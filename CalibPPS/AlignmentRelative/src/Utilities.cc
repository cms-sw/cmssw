/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "CalibPPS/AlignmentRelative/interface/Utilities.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "CalibPPS/AlignmentRelative/interface/AlignmentGeometry.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

#include "TVectorD.h"
#include "TMatrixD.h"

#include <map>
#include <set>

using namespace std;

//----------------------------------------------------------------------------------------------------

void printId(unsigned int id) {
  CTPPSDetId detId(id);

  if (detId.subdetId() == CTPPSDetId::sdTrackingStrip) {
    TotemRPDetId stDetId(id);
    printf("strip %u (%3u.%u)", id, 100 * stDetId.arm() + 10 * stDetId.station() + stDetId.rp(), stDetId.plane());
  }

  if (detId.subdetId() == CTPPSDetId::sdTimingDiamond) {
    CTPPSDiamondDetId diDetId(id);
    printf("dimnd %u (%3u.%u)", id, 100 * diDetId.arm() + 10 * diDetId.station() + diDetId.rp(), diDetId.plane());
  }

  if (detId.subdetId() == CTPPSDetId::sdTrackingPixel) {
    CTPPSPixelDetId piDetId(id);
    printf("pixel %u (%3u.%u)", id, 100 * piDetId.arm() + 10 * piDetId.station() + piDetId.rp(), piDetId.plane());
  }
}

//----------------------------------------------------------------------------------------------------

void print(TMatrixD &m, const char *label, bool mathematicaFormat) {
  if (mathematicaFormat) {
    printf("{");
    for (int i = 0; i < m.GetNrows(); i++) {
      if (i > 0)
        printf(", ");
      printf("{");
      for (int j = 0; j < m.GetNcols(); j++) {
        if (j > 0)
          printf(", ");
        printf("%.3f", m[i][j]);
      }
      printf("}");
    }
    printf("}\n");
    return;
  }

  if (label)
    printf("\n%s\n", label);

  printf("    | ");
  for (int j = 0; j < m.GetNcols(); j++)
    printf(" %9i", j);
  printf("\n------");
  for (int j = 0; j < m.GetNcols(); j++)
    printf("----------");
  printf("\n");

  for (int i = 0; i < m.GetNrows(); i++) {
    printf("%3i | ", i);
    for (int j = 0; j < m.GetNcols(); j++) {
      double v = m[i][j];
      if (fabs(v) >= 1E4)
        printf(" %+9.2E", v);
      else if (fabs(v) > 1E-6)
        printf(" %+9.2E", v);
      else
        printf("         0");
    }
    printf("\n");
  }
}

//----------------------------------------------------------------------------------------------------

void factorRPFromSensorCorrections(const CTPPSRPAlignmentCorrectionsData &inputAlignments,
                                   CTPPSRPAlignmentCorrectionsData &expandedAlignments,
                                   CTPPSRPAlignmentCorrectionsData &factoredAlignments,
                                   const AlignmentGeometry &geometry,
                                   bool equalWeights,
                                   unsigned int verbosity) {
  // clean first
  expandedAlignments.clear();
  factoredAlignments.clear();

  // save full sensor alignments
  map<unsigned int, CTPPSRPAlignmentCorrectionData> fullAlignments;
  map<unsigned int, set<unsigned int> > sensorsPerRP;
  for (auto it : inputAlignments.getSensorMap()) {
    const auto &sensorId = it.first;

    // with useRPErrors=false the only the sensor uncertainties (coming from the last analysis step) will be used
    fullAlignments[sensorId] = inputAlignments.getFullSensorCorrection(sensorId, false);

    sensorsPerRP[CTPPSDetId(sensorId).rpId()].insert(sensorId);
  }

  // convert full alignments to expandedAlignments
  for (const auto &it : fullAlignments) {
    expandedAlignments.setSensorCorrection(it.first, it.second);
  }

  // do the factorization RP per RP
  for (const auto &rpit : sensorsPerRP) {
    CTPPSDetId rpId(rpit.first);
    const set<unsigned int> &sensors = rpit.second;

    if (verbosity)
      printf("* processing RP %u (%u)\n", rpit.first, 100 * rpId.arm() + 10 * rpId.station() + rpId.rp());

    // determine number of constraints
    unsigned int n_constraints = 0;
    for (const auto &senId : sensors) {
      CTPPSDetId detId(senId);

      if (rpId.subdetId() == CTPPSDetId::sdTrackingStrip)
        n_constraints += 1;

      if (rpId.subdetId() == CTPPSDetId::sdTrackingPixel)
        n_constraints += 2;
    }

    // build matrices
    TMatrixD B(n_constraints, 2), Vi(n_constraints, n_constraints), VarM(n_constraints, n_constraints);
    TVectorD M(n_constraints);

    double sw2_sh_z = 0., svw2_sh_z = 0., su2w4_sh_z = 0.;
    double sw2_rot_x = 0., svw2_rot_x = 0., su2w4_rot_x = 0.;
    double sw2_rot_y = 0., svw2_rot_y = 0., su2w4_rot_y = 0.;
    double sw2_rot_z = 0., svw2_rot_z = 0., su2w4_rot_z = 0.;

    unsigned int idx = 0;
    for (const auto &senId : sensors) {
      CTPPSDetId detId(senId);

      const double v_sh_z = fullAlignments[senId].getShZ();
      const double u_sh_z = (fullAlignments[senId].getShZUnc() > 0.) ? fullAlignments[senId].getShZUnc() : 1.;
      const double w_sh_z = (equalWeights) ? 1. : 1. / u_sh_z;
      sw2_sh_z += w_sh_z * w_sh_z;
      svw2_sh_z += v_sh_z * w_sh_z * w_sh_z;
      su2w4_sh_z += u_sh_z * u_sh_z * w_sh_z * w_sh_z * w_sh_z * w_sh_z;

      const double v_rot_x = fullAlignments[senId].getRotX();
      const double u_rot_x = (fullAlignments[senId].getRotXUnc() > 0.) ? fullAlignments[senId].getRotXUnc() : 1.;
      const double w_rot_x = (equalWeights) ? 1. : 1. / u_rot_x;
      sw2_rot_x += w_rot_x * w_rot_x;
      svw2_rot_x += v_rot_x * w_rot_x * w_rot_x;
      su2w4_rot_x += u_rot_x * u_rot_x * w_rot_x * w_rot_x * w_rot_x * w_rot_x;

      const double v_rot_y = fullAlignments[senId].getRotY();
      const double u_rot_y = (fullAlignments[senId].getRotYUnc() > 0.) ? fullAlignments[senId].getRotYUnc() : 1.;
      const double w_rot_y = (equalWeights) ? 1. : 1. / u_rot_y;
      sw2_rot_y += w_rot_y * w_rot_y;
      svw2_rot_y += v_rot_y * w_rot_y * w_rot_y;
      su2w4_rot_y += u_rot_y * u_rot_y * w_rot_y * w_rot_y * w_rot_y * w_rot_y;

      const double v_rot_z = fullAlignments[senId].getRotZ();
      const double u_rot_z = (fullAlignments[senId].getRotZUnc() > 0.) ? fullAlignments[senId].getRotZUnc() : 1.;
      const double w_rot_z = (equalWeights) ? 1. : 1. / u_rot_z;
      sw2_rot_z += w_rot_z * w_rot_z;
      svw2_rot_z += v_rot_z * w_rot_z * w_rot_z;
      su2w4_rot_z += u_rot_z * u_rot_z * w_rot_z * w_rot_z * w_rot_z * w_rot_z;

      if (rpId.subdetId() == CTPPSDetId::sdTrackingStrip) {
        auto d2 = geometry.get(senId).getDirectionData(2);

        B(idx, 0) = d2.dx;
        B(idx, 1) = d2.dy;

        M(idx) = d2.dx * fullAlignments[senId].getShX() + d2.dy * fullAlignments[senId].getShY();
        double unc =
            sqrt(pow(d2.dx * fullAlignments[senId].getShXUnc(), 2) + pow(d2.dy * fullAlignments[senId].getShYUnc(), 2));
        if (unc <= 0.)
          unc = 1.;

        Vi(idx, idx) = (equalWeights) ? 1. : 1. / unc / unc;
        VarM(idx, idx) = unc * unc;

        idx += 1;
      }

      if (rpId.subdetId() == CTPPSDetId::sdTrackingPixel) {
        B(idx + 0, 0) = 1.;
        B(idx + 0, 1) = 0.;
        M(idx + 0) = fullAlignments[senId].getShX();
        double x_unc = fullAlignments[senId].getShXUnc();
        if (x_unc <= 0.)
          x_unc = 1.;
        Vi(idx + 0, idx + 0) = (equalWeights) ? 1. : 1. / x_unc / x_unc;
        VarM(idx + 0, idx + 0) = x_unc * x_unc;

        B(idx + 1, 0) = 0.;
        B(idx + 1, 1) = 1.;
        M(idx + 1) = fullAlignments[senId].getShY();
        double y_unc = fullAlignments[senId].getShYUnc();
        if (y_unc <= 0.)
          y_unc = 1.;
        Vi(idx + 1, idx + 1) = (equalWeights) ? 1. : 1. / y_unc / y_unc;
        VarM(idx + 1, idx + 1) = y_unc * y_unc;

        idx += 2;
      }
    }

    // calculate per-RP alignment
    TMatrixD BT(TMatrixD::kTransposed, B);
    TMatrixD BTViB(BT, TMatrixD::kMult, Vi * B);
    TMatrixD BTViBi(TMatrixD::kInverted, BTViB);
    TMatrixD S(BTViBi * BT * Vi);
    TMatrixD ST(TMatrixD::kTransposed, S);
    TVectorD th_B(2);
    th_B = S * M;
    TMatrixD VarTh_B(S * VarM * ST);

    const double m_sh_x = th_B[0], m_sh_x_unc = sqrt(VarTh_B(0, 0));
    const double m_sh_y = th_B[1], m_sh_y_unc = sqrt(VarTh_B(1, 1));
    const double m_sh_z = svw2_sh_z / sw2_sh_z, m_sh_z_unc = sqrt(su2w4_sh_z) / sw2_sh_z;

    const double m_rot_x = svw2_rot_x / sw2_rot_x, m_rot_x_unc = sqrt(su2w4_rot_x) / sw2_rot_x;
    const double m_rot_y = svw2_rot_y / sw2_rot_y, m_rot_y_unc = sqrt(su2w4_rot_y) / sw2_rot_y;
    const double m_rot_z = svw2_rot_z / sw2_rot_z, m_rot_z_unc = sqrt(su2w4_rot_z) / sw2_rot_z;

    if (verbosity) {
      printf("    m_sh_x = (%.1f +- %.1f) um, m_sh_y = (%.1f +- %.1f) um, m_sh_z = (%.1f +- %.1f) mm\n",
             m_sh_x * 1E3,
             m_sh_x_unc * 1E3,
             m_sh_y * 1E3,
             m_sh_y_unc * 1E3,
             m_sh_z,
             m_sh_z_unc);
      printf("    m_rot_x = (%.1f +- %.1f) mrad, m_rot_y = (%.1f +- %.1f)  mrad, m_rot_z = (%.1f +- %.1f) mrad\n",
             m_rot_x * 1E3,
             m_rot_x_unc * 1E3,
             m_rot_y * 1E3,
             m_rot_y_unc * 1E3,
             m_rot_z * 1E3,
             m_rot_z_unc * 1E3);
    }

    factoredAlignments.addRPCorrection(rpId,
                                       CTPPSRPAlignmentCorrectionData(m_sh_x,
                                                                      m_sh_x_unc,
                                                                      m_sh_y,
                                                                      m_sh_y_unc,
                                                                      m_sh_z,
                                                                      m_sh_z_unc,
                                                                      m_rot_x,
                                                                      m_rot_x_unc,
                                                                      m_rot_y,
                                                                      m_rot_y_unc,
                                                                      m_rot_z,
                                                                      m_rot_z_unc));

    // calculate residuals
    for (const auto &senId : sensors) {
      CTPPSRPAlignmentCorrectionData rc;

      if (rpId.subdetId() == CTPPSDetId::sdTrackingStrip) {
        auto d2 = geometry.get(senId).getDirectionData(2);

        const double de_s =
            d2.dx * (fullAlignments[senId].getShX() - m_sh_x) + d2.dy * (fullAlignments[senId].getShY() - m_sh_y);
        const double de_s_unc =
            std::abs(d2.dx * fullAlignments[senId].getShXUnc() +
                     d2.dy * fullAlignments[senId].getShYUnc());  // the x and y components are fully correlated

        rc = CTPPSRPAlignmentCorrectionData(d2.dx * de_s,
                                            d2.dx * de_s_unc,
                                            d2.dy * de_s,
                                            d2.dy * de_s_unc,
                                            fullAlignments[senId].getShZ() - m_sh_z,
                                            fullAlignments[senId].getShZUnc(),
                                            fullAlignments[senId].getRotX() - m_rot_x,
                                            fullAlignments[senId].getRotXUnc(),
                                            fullAlignments[senId].getRotY() - m_rot_y,
                                            fullAlignments[senId].getRotYUnc(),
                                            fullAlignments[senId].getRotZ() - m_rot_z,
                                            fullAlignments[senId].getRotZUnc());
      }

      if (rpId.subdetId() == CTPPSDetId::sdTrackingPixel) {
        rc = CTPPSRPAlignmentCorrectionData(fullAlignments[senId].getShX() - m_sh_x,
                                            fullAlignments[senId].getShXUnc(),
                                            fullAlignments[senId].getShY() - m_sh_y,
                                            fullAlignments[senId].getShYUnc(),
                                            fullAlignments[senId].getShZ() - m_sh_z,
                                            fullAlignments[senId].getShZUnc(),
                                            fullAlignments[senId].getRotX() - m_rot_x,
                                            fullAlignments[senId].getRotXUnc(),
                                            fullAlignments[senId].getRotY() - m_rot_y,
                                            fullAlignments[senId].getRotYUnc(),
                                            fullAlignments[senId].getRotZ() - m_rot_z,
                                            fullAlignments[senId].getRotZUnc());
      }

      factoredAlignments.addSensorCorrection(senId, rc);
    }
  }
}
