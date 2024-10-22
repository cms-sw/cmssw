/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "CalibPPS/AlignmentRelative/interface/LocalTrackFitter.h"

#include "CalibPPS/AlignmentRelative/interface/Utilities.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

#include <set>

#include "TMatrixD.h"
#include "TVectorD.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

LocalTrackFitter::LocalTrackFitter(const edm::ParameterSet &ps)
    : verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
      minimumHitsPerProjectionPerRP(ps.getParameter<unsigned int>("minimumHitsPerProjectionPerRP")),
      maxResidualToSigma(ps.getParameter<double>("maxResidualToSigma")) {}

//----------------------------------------------------------------------------------------------------

bool LocalTrackFitter::fit(HitCollection &selection, const AlignmentGeometry &geometry, LocalTrackFit &trackFit) const {
  if (verbosity > 5)
    printf(">> LocalTrackFitter::Fit\n");

  bool selectionChanged = true;
  unsigned int loopCounter = 0;
  while (selectionChanged) {
    // fit/outlier-removal loop
    while (selectionChanged) {
      if (verbosity > 5)
        printf("* fit loop %u\n", loopCounter++);

      bool fitFailed = false;
      fitAndRemoveOutliers(selection, geometry, trackFit, fitFailed, selectionChanged);

      if (fitFailed) {
        if (verbosity > 5)
          printf("\tFIT FAILED\n");
        return false;
      }
    }

    // remove pots with too few active planes
    removeInsufficientPots(selection, selectionChanged);
  }

  return true;
}

//----------------------------------------------------------------------------------------------------

void LocalTrackFitter::fitAndRemoveOutliers(HitCollection &selection,
                                            const AlignmentGeometry &geometry,
                                            LocalTrackFit &trackFit,
                                            bool &failed,
                                            bool &selectionChanged) const {
  if (verbosity > 5)
    printf(" - LocalTrackFitter::FitAndRemoveOutliers\n");

  if (selection.empty()) {
    failed = true;
    return;
  }

  // check if input size is sufficient
  if (selection.size() < 4) {
    failed = true;
    return;
  }

  // build matrices and vectors
  TMatrixD A(selection.size(), 4);
  TMatrixD Vi(selection.size(), selection.size());
  TVectorD measVec(selection.size());
  unsigned int j = 0;
  for (auto it = selection.begin(); it != selection.end(); ++it, ++j) {
    const unsigned int &detId = it->id;

    const DetGeometry &d = geometry.get(detId);
    const auto &dirData = d.getDirectionData(it->dirIdx);

    A(j, 0) = it->z * dirData.dx;
    A(j, 1) = dirData.dx;
    A(j, 2) = it->z * dirData.dy;
    A(j, 3) = dirData.dy;

    measVec(j) = it->position + dirData.s - (it->z - d.z) * dirData.dz;  // in mm

    Vi(j, j) = 1. / it->sigma / it->sigma;
  }

  // evaluate local track parameter estimates (h stands for hat)
  TMatrixD AT(4, selection.size());
  AT.Transpose(A);
  TMatrixD ATViA(4, 4);
  ATViA = AT * Vi * A;
  TMatrixD ATViAI(ATViA);

  try {
    ATViAI = ATViA.Invert();
  } catch (cms::Exception &e) {
    failed = true;
    return;
  }

  TVectorD theta(4);
  theta = ATViAI * AT * Vi * measVec;

  // residuals
  TVectorD R(measVec);
  R -= A * theta;

  // save results to trackFit
  trackFit.ax = theta(0);
  trackFit.bx = theta(1);
  trackFit.ay = theta(2);
  trackFit.by = theta(3);
  trackFit.z0 = geometry.z0;
  trackFit.ndf = selection.size() - 4;
  trackFit.chi_sq = 0;
  for (int i = 0; i < R.GetNrows(); i++)
    trackFit.chi_sq += R(i) * R(i) * Vi(i, i);

  if (verbosity > 5) {
    printf("    ax = %.3f mrad, bx = %.4f mm, ay = %.3f mrad, by = %.4f mm, z0 = %.3f mm\n",
           trackFit.ax * 1E3,
           trackFit.bx,
           trackFit.ay * 1E3,
           trackFit.by,
           trackFit.z0);
    printf("    ndof = %i, chi^2/ndof/si^2 = %.3f\n", trackFit.ndf, trackFit.chi_sq / trackFit.ndf);
  }

  // check residuals
  selectionChanged = false;
  TVectorD interpolation(A * theta);
  j = 0;
  for (auto it = selection.begin(); it != selection.end(); ++j) {
    if (verbosity > 5) {
      printf("        %2u, ", j);
      printId(it->id);
      printf(", dirIdx=%u: interpol = %+8.1f um, residual = %+6.1f um, residual / sigma = %+6.2f\n",
             it->dirIdx,
             interpolation[j] * 1E3,
             R[j] * 1E3,
             R[j] / it->sigma);
    }

    double resToSigma = R[j] / it->sigma;
    if (fabs(resToSigma) > maxResidualToSigma) {
      selection.erase(it);
      selectionChanged = true;
      if (verbosity > 5)
        printf("            Removed\n");
    } else
      ++it;
  }
}

//----------------------------------------------------------------------------------------------------

void LocalTrackFitter::removeInsufficientPots(HitCollection &selection, bool &selectionChanged) const {
  if (verbosity > 5)
    printf(" - RemoveInsufficientPots\n");

  selectionChanged = false;

  // map: RP id -> (active planes in projection 1, active planes in projection 2)
  map<unsigned int, pair<set<unsigned int>, set<unsigned int> > > planeMap;
  for (auto it = selection.begin(); it != selection.end(); ++it) {
    CTPPSDetId senId(it->id);
    const unsigned int rpId = senId.rpId();

    if (senId.subdetId() == CTPPSDetId::sdTrackingStrip) {
      const unsigned int plane = TotemRPDetId(it->id).plane();
      if ((plane % 2) == 0)
        planeMap[rpId].first.insert(senId);
      else
        planeMap[rpId].second.insert(senId);
    }

    if (senId.subdetId() == CTPPSDetId::sdTrackingPixel) {
      planeMap[rpId].first.insert(senId);
      planeMap[rpId].second.insert(senId);
    }
  }

  // remove RPs with insufficient information
  selectionChanged = false;

  for (auto it = planeMap.begin(); it != planeMap.end(); ++it) {
    if (it->second.first.size() < minimumHitsPerProjectionPerRP ||
        it->second.second.size() < minimumHitsPerProjectionPerRP) {
      if (verbosity > 5)
        printf("\tRP %u: projection1 = %lu, projection 2 = %lu\n",
               it->first,
               it->second.first.size(),
               it->second.second.size());

      // remove all hits from that RP
      for (auto dit = selection.begin(); dit != selection.end();) {
        if (it->first == CTPPSDetId(dit->id).rpId()) {
          if (verbosity > 5)
            printf("\t\tremoving %u\n", dit->id);
          selection.erase(dit);
          selectionChanged = true;
        } else {
          dit++;
        }
      }
    }
  }
}
