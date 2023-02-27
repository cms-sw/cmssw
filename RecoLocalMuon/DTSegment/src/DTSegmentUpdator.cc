/** \file
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 * \       A.Meneguzzo - Padova University  <anna.meneguzzo@pd.infn.it>
 * \       M.Pelliccioni - INFN TO <pellicci@cern.ch>
 * \       M.Meneghelli - INFN BO <marco.meneghelli@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"

/* Collaborating Class Header */

//mene
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "RecoLocalMuon/DTSegment/src/DTLinearFit.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/* C++ Headers */
#include <string>

using namespace std;
using namespace edm;

/* ====================================================================== */

/// Constructor
DTSegmentUpdator::DTSegmentUpdator(const ParameterSet& config, edm::ConsumesCollector cc)
    : theFitter{std::make_unique<DTLinearFit>()},
      theAlgo{DTRecHitAlgoFactory::get()->create(
          config.getParameter<string>("recAlgo"), config.getParameter<ParameterSet>("recAlgoConfig"), cc)},
      theGeomToken(cc.esConsumes()),
      vdrift_4parfit(config.getParameter<bool>("performT0_vdriftSegCorrection")),
      T0_hit_resolution(config.getParameter<double>("hit_afterT0_resolution")),
      perform_delta_rejecting(config.getParameter<bool>("perform_delta_rejecting")),
      debug(config.getUntrackedParameter<bool>("debug", false)) {
  intime_cut = 20.;
  if (config.exists("intime_cut"))
    intime_cut = config.getParameter<double>("intime_cut");

  if (debug)
    cout << "[DTSegmentUpdator] Constructor called" << endl;
}

/// Destructor
DTSegmentUpdator::~DTSegmentUpdator() = default;

/* Operations */

void DTSegmentUpdator::setES(const EventSetup& setup) {
  theGeom = setup.getHandle(theGeomToken);
  theAlgo->setES(setup);
}

void DTSegmentUpdator::update(DTRecSegment4D* seg, const bool calcT0, bool allow3par) const {
  if (debug)
    cout << "[DTSegmentUpdator] Starting to update the DTRecSegment4D" << endl;

  const bool hasPhi = seg->hasPhi();
  const bool hasZed = seg->hasZed();

  //reject the bad hits (due to delta rays)
  if (perform_delta_rejecting && hasPhi)
    rejectBadHits(seg->phiSegment());

  int step = (hasPhi && hasZed) ? 3 : 2;
  if (calcT0)
    step = 4;

  if (debug)
    cout << "Step of update is " << step << endl;

  GlobalPoint pos = theGeom->idToDet(seg->geographicalId())->toGlobal(seg->localPosition());
  GlobalVector dir = theGeom->idToDet(seg->geographicalId())->toGlobal(seg->localDirection());

  if (calcT0)
    calculateT0corr(seg);

  if (hasPhi)
    updateHits(seg->phiSegment(), pos, dir, step);
  if (hasZed)
    updateHits(seg->zSegment(), pos, dir, step);

  fit(seg, allow3par);
}

void DTSegmentUpdator::update(DTRecSegment2D* seg, bool allow3par) const {
  if (debug)
    cout << "[DTSegmentUpdator] Starting to update the DTRecSegment2D" << endl;
  GlobalPoint pos = (theGeom->idToDet(seg->geographicalId()))->toGlobal(seg->localPosition());
  GlobalVector dir = (theGeom->idToDet(seg->geographicalId()))->toGlobal(seg->localDirection());

  updateHits(seg, pos, dir);
  fit(seg, allow3par, false);
}

void DTSegmentUpdator::fit(DTRecSegment4D* seg, bool allow3par) const {
  if (debug)
    cout << "[DTSegmentUpdator] Fit DTRecSegment4D:" << endl;
  // after the update must refit the segments

  if (debug) {
    if (seg->hasPhi())
      cout << "    4D Segment contains a Phi segment. t0= " << seg->phiSegment()->t0()
           << "  chi2= " << seg->phiSegment()->chi2() << endl;
    if (seg->hasZed())
      cout << "    4D Segment contains a Zed segment. t0= " << seg->zSegment()->t0()
           << "  chi2= " << seg->zSegment()->chi2() << endl;
  }

  // If both phi and zed projections are present and the phi segment is in time (segment t0<intime_cut) the 3-par fit is blocked and
  // segments are fit with the 2-par fit. Setting intime_cut to -1 results in the 3-par fit being used always.
  if (seg->hasPhi()) {
    if (seg->hasZed()) {
      if (fabs(seg->phiSegment()->t0()) < intime_cut) {
        fit(seg->phiSegment(), allow3par, true);
        fit(seg->zSegment(), allow3par, true);
      } else {
        fit(seg->phiSegment(), allow3par, false);
        fit(seg->zSegment(), allow3par, false);
      }
    } else
      fit(seg->phiSegment(), allow3par, false);
  } else
    fit(seg->zSegment(), allow3par, false);

  const DTChamber* theChamber = theGeom->chamber(seg->chamberId());

  if (seg->hasPhi() && seg->hasZed()) {
    DTChamberRecSegment2D* segPhi = seg->phiSegment();
    DTSLRecSegment2D* segZed = seg->zSegment();

    // NB Phi seg is already in chamber ref
    LocalPoint posPhiInCh = segPhi->localPosition();
    LocalVector dirPhiInCh = segPhi->localDirection();

    // Zed seg is in SL one
    const DTSuperLayer* zSL = theChamber->superLayer(segZed->superLayerId());
    LocalPoint zPos(segZed->localPosition().x(), (zSL->toLocal(theChamber->toGlobal(segPhi->localPosition()))).y(), 0.);

    LocalPoint posZInCh = theChamber->toLocal(zSL->toGlobal(zPos));

    LocalVector dirZInCh = theChamber->toLocal(zSL->toGlobal(segZed->localDirection()));

    LocalPoint posZAt0 = posZInCh + dirZInCh * (-posZInCh.z()) / cos(dirZInCh.theta());

    // given the actual definition of chamber refFrame, (with z poiniting to IP),
    // the zed component of direction is negative.
    LocalVector dir = LocalVector(dirPhiInCh.x() / fabs(dirPhiInCh.z()), dirZInCh.y() / fabs(dirZInCh.z()), -1.);

    seg->setPosition(LocalPoint(posPhiInCh.x(), posZAt0.y(), 0.));
    seg->setDirection(dir.unit());

    AlgebraicSymMatrix mat(4);

    // set cov matrix
    mat[0][0] = segPhi->parametersError()[0][0];  //sigma (dx/dz)
    mat[0][2] = segPhi->parametersError()[0][1];  //cov(dx/dz,x)
    mat[2][2] = segPhi->parametersError()[1][1];  //sigma (x)

    seg->setCovMatrix(mat);
    seg->setCovMatrixForZed(posZInCh);
  } else if (seg->hasPhi()) {
    DTChamberRecSegment2D* segPhi = seg->phiSegment();

    seg->setPosition(segPhi->localPosition());
    seg->setDirection(segPhi->localDirection());

    AlgebraicSymMatrix mat(4);
    // set cov matrix
    mat[0][0] = segPhi->parametersError()[0][0];  //sigma (dx/dz)
    mat[0][2] = segPhi->parametersError()[0][1];  //cov(dx/dz,x)
    mat[2][2] = segPhi->parametersError()[1][1];  //sigma (x)

    seg->setCovMatrix(mat);
  } else if (seg->hasZed()) {
    DTSLRecSegment2D* segZed = seg->zSegment();

    // Zed seg is in SL one
    GlobalPoint glbPosZ = (theGeom->superLayer(segZed->superLayerId()))->toGlobal(segZed->localPosition());
    LocalPoint posZInCh = (theGeom->chamber(segZed->superLayerId().chamberId()))->toLocal(glbPosZ);

    GlobalVector glbDirZ = (theGeom->superLayer(segZed->superLayerId()))->toGlobal(segZed->localDirection());
    LocalVector dirZInCh = (theGeom->chamber(segZed->superLayerId().chamberId()))->toLocal(glbDirZ);

    LocalPoint posZAt0 = posZInCh + dirZInCh * (-posZInCh.z()) / cos(dirZInCh.theta());

    seg->setPosition(posZAt0);
    seg->setDirection(dirZInCh);

    AlgebraicSymMatrix mat(4);
    // set cov matrix
    seg->setCovMatrix(mat);
    seg->setCovMatrixForZed(posZInCh);
  }
}

bool DTSegmentUpdator::fit(DTSegmentCand* seg, bool allow3par, const bool fitdebug) const {
  //  if (debug && fitdebug) cout << "[DTSegmentUpdator] Fit DTRecSegment2D" << endl;
  //  if (!seg->good()) return false;

  //  DTSuperLayerId DTid = (DTSuperLayerId)seg->superLayer()->id();
  //  if (DTid.superlayer()==2)
  //    allow3par = 0;

  vector<float> x;
  vector<float> y;
  vector<float> sigy;
  vector<int> lfit;
  vector<double> dist;
  int i = 0;

  x.reserve(8);
  y.reserve(8);
  sigy.reserve(8);
  lfit.reserve(8);
  dist.reserve(8);

  for (DTSegmentCand::AssPointCont::const_iterator iter = seg->hits().begin(); iter != seg->hits().end(); ++iter) {
    LocalPoint pos = (*iter).first->localPosition((*iter).second);
    float xwire =
        (((*iter).first)->localPosition(DTEnums::Left).x() + ((*iter).first)->localPosition(DTEnums::Right).x()) / 2.;
    float distance = pos.x() - xwire;

    if ((*iter).second == DTEnums::Left)
      lfit.push_back(1);
    else
      lfit.push_back(-1);

    dist.push_back(distance);
    sigy.push_back(sqrt((*iter).first->localPositionError().xx()));
    x.push_back(pos.z());
    y.push_back(pos.x());
    i++;
  }

  LocalPoint pos;
  LocalVector dir;
  AlgebraicSymMatrix covMat(2);
  float cminf = 0.;
  float vminf = 0.;
  double chi2 = 0.;
  double t0_corr = 0.;

  fit(x, y, lfit, dist, sigy, pos, dir, cminf, vminf, covMat, chi2, allow3par);
  if (cminf != 0)
    t0_corr = -cminf / 0.00543;  // convert drift distance to time

  if (debug && fitdebug)
    cout << "  DTcand chi2: " << chi2 << "/" << x.size() << "   t0: " << t0_corr << endl;

  seg->setPosition(pos);
  seg->setDirection(dir);
  seg->sett0(t0_corr);
  seg->setCovMatrix(covMat);
  seg->setChi2(chi2);

  // cout << "pos " << pos << endl;
  // cout << "dir " << dir << endl;
  // cout << "Mat " << covMat << endl;

  return true;
}

void DTSegmentUpdator::fit(DTRecSegment2D* seg, bool allow3par, bool block3par) const {
  if (debug)
    cout << "[DTSegmentUpdator] Fit DTRecSegment2D - 3par: " << allow3par << endl;

  vector<float> x;
  vector<float> y;
  vector<float> sigy;
  vector<int> lfit;
  vector<double> dist;
  x.reserve(8);
  y.reserve(8);
  sigy.reserve(8);
  lfit.reserve(8);
  dist.reserve(8);

  //  DTSuperLayerId DTid = (DTSuperLayerId)seg->geographicalId();
  //  if (DTid.superlayer()==2)
  //    allow3par = 0;

  vector<DTRecHit1D> hits = seg->specificRecHits();
  for (vector<DTRecHit1D>::const_iterator hit = hits.begin(); hit != hits.end(); ++hit) {
    // I have to get the hits position (the hit is in the layer rf) in SL frame...
    GlobalPoint glbPos = (theGeom->layer(hit->wireId().layerId()))->toGlobal(hit->localPosition());
    LocalPoint pos = (theGeom->idToDet(seg->geographicalId()))->toLocal(glbPos);
    x.push_back(pos.z());
    y.push_back(pos.x());

    const DTLayer* layer = theGeom->layer(hit->wireId().layerId());
    float xwire = layer->specificTopology().wirePosition(hit->wireId().wire());
    float distance = fabs(hit->localPosition().x() - xwire);
    dist.push_back(distance);

    int ilc = (hit->lrSide() == DTEnums::Left) ? 1 : -1;
    lfit.push_back(ilc);

    // Get local error in SL frame
    //RB: is it right in this way?
    ErrorFrameTransformer tran;
    GlobalError glbErr =
        tran.transform(hit->localPositionError(), (theGeom->layer(hit->wireId().layerId()))->surface());
    LocalError slErr = tran.transform(glbErr, (theGeom->idToDet(seg->geographicalId()))->surface());
    sigy.push_back(sqrt(slErr.xx()));
  }

  LocalPoint pos;
  LocalVector dir;
  AlgebraicSymMatrix covMat(2);
  double chi2 = 0.;
  float cminf = 0.;
  float vminf = 0.;
  double t0_corr = 0.;

  fit(x, y, lfit, dist, sigy, pos, dir, cminf, vminf, covMat, chi2, allow3par, block3par);
  if (cminf != 0)
    t0_corr = -cminf / 0.00543;  // convert drift distance to time

  if (debug)
    cout << "   DTSeg2d chi2: " << chi2 << endl;
  if (debug)
    cout << "   DTSeg2d Fit t0: " << t0_corr << endl;
  // cout << "pos " << segPosition << endl;
  // cout << "dir " << segDirection << endl;
  // cout << "Mat " << mat << endl;

  seg->setPosition(pos);
  seg->setDirection(dir);
  seg->setCovMatrix(covMat);
  seg->setChi2(chi2);
  seg->setT0(t0_corr);
}

void DTSegmentUpdator::fit(const vector<float>& x,
                           const vector<float>& y,
                           const vector<int>& lfit,
                           const vector<double>& dist,
                           const vector<float>& sigy,
                           LocalPoint& pos,
                           LocalVector& dir,
                           float& cminf,
                           float& vminf,
                           AlgebraicSymMatrix& covMatrix,
                           double& chi2,
                           const bool allow3par,
                           const bool block3par) const {
  float slope = 0.;
  float intercept = 0.;
  float covss = 0.;
  float covii = 0.;
  float covsi = 0.;

  cminf = 0;
  vminf = 0;

  int leftHits = 0, rightHits = 0;
  for (unsigned int i = 0; i < lfit.size(); i++)
    if (lfit[i] == 1)
      leftHits++;
    else
      rightHits++;

  theFitter->fit(x, y, x.size(), sigy, slope, intercept, chi2, covss, covii, covsi);

  // If we have at least one left and one right hit we can try the 3 parameter fit (if it is switched on)
  // FIXME: currently the covariance matrix from the 2-par fit is kept
  if (leftHits && rightHits && (leftHits + rightHits > 3) && allow3par) {
    theFitter->fitNpar(3, x, y, lfit, dist, sigy, slope, intercept, cminf, vminf, chi2, debug);
    double t0_corr = -cminf / 0.00543;
    if (fabs(t0_corr) < intime_cut && block3par) {
      theFitter->fit(x, y, x.size(), sigy, slope, intercept, chi2, covss, covii, covsi);
      cminf = 0;
    }
  }

  // cout << "slope " << slope << endl;
  // cout << "intercept " << intercept << endl;

  // intercept is the x() in chamber frame when the segment cross the chamber
  // plane (at z()=0), the y() is not measured, so let's put the center of the
  // chamber.
  pos = LocalPoint(intercept, 0., 0.);

  //  slope is dx()/dz(), while dy()/dz() is by definition 0, finally I want the
  //  segment to point outward, so opposite to local z
  dir = LocalVector(-slope, 0., -1.).unit();

  covMatrix = AlgebraicSymMatrix(2);
  covMatrix[0][0] = covss;  // this is var(dy/dz)
  covMatrix[1][1] = covii;  // this is var(y)
  covMatrix[1][0] = covsi;  // this is cov(dy/dz,y)
}

// The GlobalPoint and the GlobalVector can be either the glb position and the direction
// of the 2D-segment itself or the glb position and direction of the 4D segment
void DTSegmentUpdator::updateHits(DTRecSegment2D* seg, GlobalPoint& gpos, GlobalVector& gdir, const int step) const {
  // it is not necessary to have DTRecHit1D* to modify the obj in the container
  // but I have to be carefully, since I cannot make a copy before the iteration!

  vector<DTRecHit1D> toBeUpdatedRecHits = seg->specificRecHits();
  vector<DTRecHit1D> updatedRecHits;

  for (vector<DTRecHit1D>::iterator hit = toBeUpdatedRecHits.begin(); hit != toBeUpdatedRecHits.end(); ++hit) {
    const DTLayer* layer = theGeom->layer(hit->wireId().layerId());

    LocalPoint segPos = layer->toLocal(gpos);
    LocalVector segDir = layer->toLocal(gdir);

    // define impact angle needed by the step 2
    const float angle = atan(segDir.x() / -segDir.z());

    // define the local position (extr.) of the segment. Needed by the third step
    LocalPoint segPosAtLayer = segPos + segDir * (-segPos.z()) / cos(segDir.theta());

    DTRecHit1D newHit1D = (*hit);
    bool ok = true;

    if (step == 2) {
      ok = theAlgo->compute(layer, *hit, angle, newHit1D);

    } else if (step == 3) {
      LocalPoint hitPos(hit->localPosition().x(), +segPosAtLayer.y(), 0.);
      GlobalPoint glbpos = theGeom->layer(hit->wireId().layerId())->toGlobal(hitPos);
      newHit1D.setPosition(hitPos);
      ok = theAlgo->compute(layer, *hit, angle, glbpos, newHit1D);

    } else if (step == 4) {
      //const double vminf = seg->vDrift();   //  vdrift correction are recorded in the segment
      double vminf = 0.;
      if (vdrift_4parfit)
        vminf = seg->vDrift();  // use vdrift recorded in the segment only if vdrift_4parfit=True

      double cminf = 0.;
      if (seg->ist0Valid())
        cminf = -seg->t0() * 0.00543;

      //cout << "In updateHits: t0 = " << seg->t0() << endl;
      //cout << "In updateHits: vminf = " << vminf << endl;
      //cout << "In updateHits: cminf = " << cminf << endl;

      const float xwire = layer->specificTopology().wirePosition(hit->wireId().wire());
      const float distance = fabs(hit->localPosition().x() - xwire);
      const int ilc = (hit->lrSide() == DTEnums::Left) ? 1 : -1;
      const double dy_corr = (vminf * ilc * distance - cminf * ilc);

      LocalPoint point(hit->localPosition().x() + dy_corr, +segPosAtLayer.y(), 0.);

      //double final_hit_resol = T0_hit_resolution;
      //if(newHit1D.wireId().layerId().superlayerId().superLayer() != 2) final_hit_resol = final_hit_resol * 0.8;
      //LocalError error(final_hit_resol * final_hit_resol,0.,0.);
      LocalError error(T0_hit_resolution * T0_hit_resolution, 0., 0.);
      newHit1D.setPositionAndError(point, error);

      //FIXME: check that the hit is still inside the cell
      ok = true;

    } else
      throw cms::Exception("DTSegmentUpdator") << " updateHits called with wrong step " << endl;

    if (ok)
      updatedRecHits.push_back(newHit1D);
    else {
      LogError("DTSegmentUpdator") << "DTSegmentUpdator::updateHits failed update" << endl;
      throw cms::Exception("DTSegmentUpdator") << "updateHits failed update" << endl;
    }
  }
  seg->update(updatedRecHits);
}

void DTSegmentUpdator::rejectBadHits(DTChamberRecSegment2D* phiSeg) const {
  vector<float> x;
  vector<float> y;

  if (debug)
    cout << " Inside the segment updator, now loop on hits:   ( x == z_loc , y == x_loc) " << endl;

  vector<DTRecHit1D> hits = phiSeg->specificRecHits();
  const size_t N = hits.size();
  if (N < 3)
    return;

  for (vector<DTRecHit1D>::const_iterator hit = hits.begin(); hit != hits.end(); ++hit) {
    // I have to get the hits position (the hit is in the layer rf) in SL frame...
    GlobalPoint glbPos = (theGeom->layer(hit->wireId().layerId()))->toGlobal(hit->localPosition());
    LocalPoint pos = (theGeom->idToDet(phiSeg->geographicalId()))->toLocal(glbPos);

    x.push_back(pos.z());
    y.push_back(pos.x());
  }

  if (debug) {
    cout << " end of segment! " << endl;
    cout << " size = Number of Hits: " << x.size() << "  " << y.size() << endl;
  }

  // Perform the 2 par fit:
  float par[2] = {0., 0.};  // q , m

  //variables to perform the fit:
  float Sx = 0.;
  float Sy = 0.;
  float Sx2 = 0.;
  float Sxy = 0.;

  for (size_t i = 0; i < N; ++i) {
    Sx += x.at(i);
    Sy += y.at(i);
    Sx2 += x.at(i) * x.at(i);
    Sxy += x.at(i) * y.at(i);
  }

  const float delta = N * Sx2 - Sx * Sx;
  par[0] = (Sx2 * Sy - Sx * Sxy) / delta;
  par[1] = (N * Sxy - Sx * Sy) / delta;

  if (debug)
    cout << "fit 2 parameters done ----> par0: " << par[0] << "  par1: " << par[1] << endl;

  // Calc residuals:
  float residuals[N];
  float mean_residual = 0.;  //mean of the absolute values of residuals
  for (size_t i = 0; i < N; ++i) {
    residuals[i] = y.at(i) - par[1] * x.at(i) - par[0];
    mean_residual += std::abs(residuals[i]);
    if (debug) {
      cout << " i: " << i << " y_i " << y.at(i) << " x_i " << x.at(i) << " res_i " << residuals[i];
      if (i == N - 1)
        cout << endl;
    }
  }

  if (debug)
    cout << " Residuals computed! " << endl;

  mean_residual = mean_residual / (N - 2);
  if (debug)
    cout << " mean_residual: " << mean_residual << endl;

  int i = 0;

  // Perform bad hit rejecting -- update hits
  vector<DTRecHit1D> updatedRecHits;
  for (vector<DTRecHit1D>::const_iterator hit = hits.begin(); hit != hits.end(); ++hit) {
    float normResidual = mean_residual > 0 ? std::abs(residuals[i]) / mean_residual : 0;
    ++i;
    if (normResidual < 1.5) {
      const DTRecHit1D& newHit1D = (*hit);
      updatedRecHits.push_back(newHit1D);
      if (debug)
        cout << " accepted " << i << "th hit"
             << "  Irej: " << normResidual << endl;
    } else {
      if (debug)
        cout << " rejected " << i << "th hit"
             << "  Irej: " << normResidual << endl;
      continue;
    }
  }

  phiSeg->update(updatedRecHits);

  //final check!
  if (debug) {
    vector<float> x_upd;
    vector<float> y_upd;

    cout << " Check the update action: " << endl;

    vector<DTRecHit1D> hits_upd = phiSeg->specificRecHits();
    for (vector<DTRecHit1D>::const_iterator hit = hits_upd.begin(); hit != hits_upd.end(); ++hit) {
      // I have to get the hits position (the hit is in the layer rf) in SL frame...
      GlobalPoint glbPos = (theGeom->layer(hit->wireId().layerId()))->toGlobal(hit->localPosition());
      LocalPoint pos = (theGeom->idToDet(phiSeg->geographicalId()))->toLocal(glbPos);

      x_upd.push_back(pos.z());
      y_upd.push_back(pos.x());

      cout << " x_upd: " << pos.z() << "  y_upd: " << pos.x() << endl;
    }

    cout << " end of segment! " << endl;
    cout << " size = Number of Hits: " << x_upd.size() << "  " << y_upd.size() << endl;

  }  // end debug

  return;
}  //end DTSegmentUpdator::rejectBadHits

void DTSegmentUpdator::calculateT0corr(DTRecSegment4D* seg) const {
  if (seg->hasPhi())
    calculateT0corr(seg->phiSegment());
  if (seg->hasZed())
    calculateT0corr(seg->zSegment());
}

void DTSegmentUpdator::calculateT0corr(DTRecSegment2D* seg) const {
  // WARNING: since this method is called both with a 2D and a 2DPhi as argument
  // seg->geographicalId() can be a superLayerId or a chamberId
  if (debug)
    cout << "[DTSegmentUpdator] CalculateT0corr DTRecSegment4D" << endl;

  vector<double> d_drift;
  vector<float> x;
  vector<float> y;
  vector<int> lc;

  vector<DTRecHit1D> hits = seg->specificRecHits();

  DTWireId wireId;
  int nptfit = 0;

  for (vector<DTRecHit1D>::const_iterator hit = hits.begin(); hit != hits.end(); ++hit) {
    // I have to get the hits position (the hit is in the layer rf) in SL frame...
    GlobalPoint glbPos = (theGeom->layer(hit->wireId().layerId()))->toGlobal(hit->localPosition());
    LocalPoint pos = (theGeom->idToDet(seg->geographicalId()))->toLocal(glbPos);

    const DTLayer* layer = theGeom->layer(hit->wireId().layerId());
    float xwire = layer->specificTopology().wirePosition(hit->wireId().wire());
    float distance = fabs(hit->localPosition().x() - xwire);

    int ilc = (hit->lrSide() == DTEnums::Left) ? 1 : -1;

    nptfit++;
    x.push_back(pos.z());
    y.push_back(pos.x());
    lc.push_back(ilc);
    d_drift.push_back(distance);

    // cout << " d_drift "<<distance  <<" npt= " <<npt<<endl;
  }

  double chi2fit = 0.;
  float cminf = 0.;
  float vminf = 0.;
  float a, b;

  if (nptfit > 2) {
    //NB chi2fit is normalized
    theFitter->fit4Var(x, y, lc, d_drift, nptfit, a, b, cminf, vminf, chi2fit, vdrift_4parfit, debug);

    double t0cor = -999.;
    if (cminf > -998.)
      t0cor = -cminf / 0.00543;  // in ns

    //cout << "In calculateT0corr: t0 = " << t0cor << endl;
    //cout << "In calculateT0corr: vminf = " << vminf << endl;
    //cout << "In calculateT0corr: cminf = " << cminf << endl;
    //cout << "In calculateT0corr: chi2 = " << chi2fit << endl;

    seg->setT0(t0cor);      // time  and
    seg->setVdrift(vminf);  //  vdrift correction are recorded in the segment
  }
}
