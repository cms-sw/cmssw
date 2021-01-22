#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithm.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit2D.h"

void VectorHitBuilderAlgorithm::run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                                    VectorHitCollection& vhAcc,
                                    VectorHitCollection& vhRej,
                                    edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
                                    edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej) const {
  LogDebug("VectorHitBuilderAlgorithm") << "Run VectorHitBuilderAlgorithm ... \n";
  const auto* clustersPhase2Collection = clusters.product();

  //loop over the DetSetVector
  LogDebug("VectorHitBuilderAlgorithm") << "with #clusters : " << clustersPhase2Collection->size() << std::endl;
  for (auto dSViter : *clustersPhase2Collection) {
    unsigned int rawDetId1(dSViter.detId());
    DetId detId1(rawDetId1);
    DetId lowerDetId, upperDetId;
    if (tkTopo_->isLower(detId1)) {
      lowerDetId = detId1;
      upperDetId = tkTopo_->partnerDetId(detId1);
    } else
      continue;

    DetId detIdStack = tkTopo_->stack(detId1);

    //debug
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId stack : " << detIdStack.rawId() << std::endl;
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId lower set of clusters  : " << lowerDetId.rawId();
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId upper set of clusters  : " << upperDetId.rawId() << std::endl;

    const GeomDet* gd;
    const StackGeomDet* stackDet;
    const auto& it_detLower = dSViter;
    const auto& it_detUpper = clustersPhase2Collection->find(upperDetId);

    if (it_detUpper != clustersPhase2Collection->end()) {
      gd = tkGeom_->idToDet(detIdStack);
      stackDet = dynamic_cast<const StackGeomDet*>(gd);
      buildVectorHits(vhAcc, vhRej, detIdStack, stackDet, clusters, it_detLower, *it_detUpper);
    }
  }
  LogDebug("VectorHitBuilderAlgorithm") << "End run VectorHitBuilderAlgorithm ... \n";
}

bool VectorHitBuilderAlgorithm::checkClustersCompatibilityBeforeBuilding(
    edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
    const Detset& theLowerDetSet,
    const Detset& theUpperDetSet) const {
  if (theLowerDetSet.size() == 1 && theUpperDetSet.size() == 1)
    return true;

  //order lower clusters in u
  std::vector<Phase2TrackerCluster1D> lowerClusters;
  lowerClusters.reserve(theLowerDetSet.size());
  if (theLowerDetSet.size() > 1)
    LogDebug("VectorHitBuilderAlgorithm") << " more than 1 lower cluster! " << std::endl;
  if (theUpperDetSet.size() > 1)
    LogDebug("VectorHitBuilderAlgorithm") << " more than 1 upper cluster! " << std::endl;
  for (const_iterator cil = theLowerDetSet.begin(); cil != theLowerDetSet.end(); ++cil) {
    Phase2TrackerCluster1DRef clusterLower = edmNew::makeRefTo(clusters, cil);
    lowerClusters.push_back(*clusterLower);
  }
  return true;
}

bool VectorHitBuilderAlgorithm::checkClustersCompatibility(Local3DPoint& poslower,
                                                           Local3DPoint& posupper,
                                                           LocalError& errlower,
                                                           LocalError& errupper) const {
  return true;
}

//----------------------------------------------------------------------------
//ERICA::in the DT code the global position is used to compute the alpha angle and put a cut on that.
void VectorHitBuilderAlgorithm::buildVectorHits(VectorHitCollection& vhAcc,
                                                VectorHitCollection& vhRej,
                                                DetId detIdStack,
                                                const StackGeomDet* stack,
                                                edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                                                const Detset& theLowerDetSet,
                                                const Detset& theUpperDetSet,
                                                const std::vector<bool>& phase2OTClustersToSkip) const {
  if (checkClustersCompatibilityBeforeBuilding(clusters, theLowerDetSet, theUpperDetSet)) {
    LogDebug("VectorHitBuilderAlgorithm") << "  compatible -> continue ... " << std::endl;
  } else {
    LogTrace("VectorHitBuilderAlgorithm") << "  not compatible, going to the next cluster";
  }

  edmNew::DetSetVector<VectorHit>::FastFiller vh_colAcc(vhAcc, detIdStack);
  edmNew::DetSetVector<VectorHit>::FastFiller vh_colRej(vhRej, detIdStack);

  unsigned int layerStack = tkTopo_->layer(stack->geographicalId());
  if (stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTB)
    LogDebug("VectorHitBuilderAlgorithm") << " \t is barrel.    " << std::endl;
  if (stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC)
    LogDebug("VectorHitBuilderAlgorithm") << " \t is endcap.    " << std::endl;
  LogDebug("VectorHitBuilderAlgorithm") << " \t layer is : " << layerStack << std::endl;

  float cut = 0.0;
  if (stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTB)
    cut = barrelCut_.at(layerStack);
  if (stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC)
    cut = endcapCut_.at(layerStack);
  LogDebug("VectorHitBuilderAlgorithm") << " \t the cut is:" << cut << std::endl;

  //only cache local parameters for upper cluster as we loop over lower clusters only once anyway
  std::vector<std::pair<LocalPoint, LocalError>> localParamsUpper;
  std::vector<const PixelGeomDetUnit*> localGDUUpper;

  const PixelGeomDetUnit* gduUpp = dynamic_cast<const PixelGeomDetUnit*>(stack->upperDet());
  for (auto const& clusterUpper : theUpperDetSet) {
    localGDUUpper.push_back(gduUpp);
    localParamsUpper.push_back(cpe_->localParameters(clusterUpper, *gduUpp));
  }
  int upperIterator = 0;
  const PixelGeomDetUnit* gduLow = dynamic_cast<const PixelGeomDetUnit*>(stack->lowerDet());
  for (const_iterator cil = theLowerDetSet.begin(); cil != theLowerDetSet.end(); ++cil) {
    LogDebug("VectorHitBuilderAlgorithm") << " lower clusters " << std::endl;
    Phase2TrackerCluster1DRef cluL = edmNew::makeRefTo(clusters, cil);
#ifdef EDM_ML_DEBUG
    printCluster(stack->lowerDet(), &*cluL);
#endif
    auto&& lparamsLow = cpe_->localParameters(*cluL, *gduLow);
    upperIterator = 0;
    for (const_iterator ciu = theUpperDetSet.begin(); ciu != theUpperDetSet.end(); ++ciu) {
      LogDebug("VectorHitBuilderAlgorithm") << "\t upper clusters " << std::endl;
      Phase2TrackerCluster1DRef cluU = edmNew::makeRefTo(clusters, ciu);
#ifdef EDM_ML_DEBUG
      printCluster(stack->upperDet(), &*cluU);
#endif
      //applying the parallax correction
      double pC = computeParallaxCorrection(
          gduLow, lparamsLow.first, localGDUUpper[upperIterator], localParamsUpper[upperIterator].first);
      LogDebug("VectorHitBuilderAlgorithm") << " \t parallax correction:" << pC << std::endl;
      double lpos_upp_corr = 0.0;
      double lpos_low_corr = 0.0;
      auto const localUpperX = localParamsUpper[upperIterator].first.x();
      if (localUpperX > lparamsLow.first.x()) {
        if (localUpperX > 0) {
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = localParamsUpper[upperIterator].first.x() - std::abs(pC);
        } else if (localUpperX < 0) {
          lpos_low_corr = lparamsLow.first.x() + std::abs(pC);
          lpos_upp_corr = localUpperX;
        }
      } else if (localUpperX < lparamsLow.first.x()) {
        if (localUpperX > 0) {
          lpos_low_corr = lparamsLow.first.x() - std::abs(pC);
          lpos_upp_corr = localUpperX;
        } else if (localUpperX < 0) {
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = localUpperX + std::abs(pC);
        }
      } else {
        if (localUpperX > 0) {
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = localUpperX - std::abs(pC);
        } else if (localUpperX < 0) {
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = localUpperX + std::abs(pC);
        }
      }

      LogDebug("VectorHitBuilderAlgorithm") << " \t local pos upper corrected (x):" << lpos_upp_corr << std::endl;
      LogDebug("VectorHitBuilderAlgorithm") << " \t local pos lower corrected (x):" << lpos_low_corr << std::endl;

      double width = lpos_low_corr - lpos_upp_corr;
      LogDebug("VectorHitBuilderAlgorithm") << " \t width: " << width << std::endl;

      //old cut: indipendent from layer
      //building my tolerance : 10*sigma
      //double delta = 10.0 * sqrt(lparamsLow.second.xx() + localParamsUpper[upperIterator].second.xx());
      //LogDebug("VectorHitBuilderAlgorithm") << " \t delta: " << delta << std::endl;
      //if( (lpos_upp_corr < lpos_low_corr + delta) &&
      //    (lpos_upp_corr > lpos_low_corr - delta) ){
      //new cut: dependent on layers
      if (std::abs(width) < cut) {
        LogDebug("VectorHitBuilderAlgorithm") << " accepting VH! " << std::endl;
        VectorHit vh = buildVectorHit(stack, cluL, cluU);
        //protection: the VH can also be empty!!
        if (vh.isValid()) {
          vh_colAcc.push_back(vh);
        }

      } else {
        LogDebug("VectorHitBuilderAlgorithm") << " rejecting VH: " << std::endl;
        //storing vh rejected for combinatiorial studies
        VectorHit vh = buildVectorHit(stack, cluL, cluU);
        if (vh.isValid()) {
          vh_colRej.push_back(vh);
        }
      }
      upperIterator += 1;
    }
  }
}

VectorHit VectorHitBuilderAlgorithm::buildVectorHit(const StackGeomDet* stack,
                                                    Phase2TrackerCluster1DRef lower,
                                                    Phase2TrackerCluster1DRef upper) const {
  LogTrace("VectorHitBuilderAlgorithm") << "Build VH with: ";
#ifdef EDM_ML_DEBUG
  printCluster(stack->upperDet(), &*upper);
#endif
  const PixelGeomDetUnit* geomDetLower = static_cast<const PixelGeomDetUnit*>(stack->lowerDet());
  const PixelGeomDetUnit* geomDetUpper = static_cast<const PixelGeomDetUnit*>(stack->upperDet());

  auto&& lparamsLower = cpe_->localParameters(*lower, *geomDetLower);  // x, y, z, e2_xx, e2_xy, e2_yy
  Global3DPoint gparamsLower = geomDetLower->surface().toGlobal(lparamsLower.first);
  LogTrace("VectorHitBuilderAlgorithm") << "\t lower global pos: " << gparamsLower;

  auto&& lparamsUpper = cpe_->localParameters(*upper, *geomDetUpper);
  Global3DPoint gparamsUpper = geomDetUpper->surface().toGlobal(lparamsUpper.first);
  LogTrace("VectorHitBuilderAlgorithm") << "\t upper global pos: " << gparamsUpper;

  //local parameters of upper cluster in lower system of reference
  Local3DPoint lparamsUpperInLower = geomDetLower->surface().toLocal(gparamsUpper);

  LogTrace("VectorHitBuilderAlgorithm") << "\t lower global pos: " << gparamsLower;
  LogTrace("VectorHitBuilderAlgorithm") << "\t upper global pos: " << gparamsUpper;

  LogTrace("VectorHitBuilderAlgorithm") << "A:\t lower local pos: " << lparamsLower.first
                                        << " with error: " << lparamsLower.second << std::endl;
  LogTrace("VectorHitBuilderAlgorithm") << "A:\t upper local pos in the lower sof " << lparamsUpperInLower
                                        << " with error: " << lparamsUpper.second << std::endl;

  bool ok =
      checkClustersCompatibility(lparamsLower.first, lparamsUpper.first, lparamsLower.second, lparamsUpper.second);

  if (ok) {
    AlgebraicSymMatrix22 covMat2Dzx;
    double chi22Dzx = 0.0;
    Local3DPoint pos2Dzx;
    Local3DVector dir2Dzx;
    fit2Dzx(lparamsLower.first,
            lparamsUpperInLower,
            lparamsLower.second,
            lparamsUpper.second,
            pos2Dzx,
            dir2Dzx,
            covMat2Dzx,
            chi22Dzx);
    LogTrace("VectorHitBuilderAlgorithm") << "\t  pos2Dzx: " << pos2Dzx;
    LogTrace("VectorHitBuilderAlgorithm") << "\t  dir2Dzx: " << dir2Dzx;
    LogTrace("VectorHitBuilderAlgorithm") << "\t  cov2Dzx: " << covMat2Dzx;
    VectorHit2D vh2Dzx = VectorHit2D(pos2Dzx, dir2Dzx, covMat2Dzx, chi22Dzx);

    AlgebraicSymMatrix22 covMat2Dzy;
    double chi22Dzy = 0.0;
    Local3DPoint pos2Dzy;
    Local3DVector dir2Dzy;
    fit2Dzy(lparamsLower.first,
            lparamsUpperInLower,
            lparamsLower.second,
            lparamsUpper.second,
            pos2Dzy,
            dir2Dzy,
            covMat2Dzy,
            chi22Dzy);
    LogTrace("VectorHitBuilderAlgorithm") << "\t  pos2Dzy: " << pos2Dzy;
    LogTrace("VectorHitBuilderAlgorithm") << "\t  dir2Dzy: " << dir2Dzy;
    LogTrace("VectorHitBuilderAlgorithm") << "\t  cov2Dzy: " << covMat2Dzy;
    VectorHit2D vh2Dzy = VectorHit2D(pos2Dzy, dir2Dzy, covMat2Dzy, chi22Dzy);

    OmniClusterRef lowerOmni(lower);
    OmniClusterRef upperOmni(upper);

    Global3DPoint gPositionLower = VectorHit::phase2clusterGlobalPos(geomDetLower, lower);
    Global3DPoint gPositionUpper = VectorHit::phase2clusterGlobalPos(geomDetUpper, upper);
    GlobalError gErrorLower = VectorHit::phase2clusterGlobalPosErr(geomDetLower);
    GlobalError gErrorUpper = VectorHit::phase2clusterGlobalPosErr(geomDetUpper);

    if (gPositionLower.perp() > gPositionUpper.perp()) {
      std::swap(gPositionLower, gPositionUpper);
      std::swap(gErrorLower, gErrorUpper);
    }

    const CurvatureAndPhi curvatureAndPhi = curvatureANDphi(gPositionLower, gPositionUpper, gErrorLower, gErrorUpper);
    VectorHit vh = VectorHit(*stack,
                             vh2Dzx,
                             vh2Dzy,
                             lowerOmni,
                             upperOmni,
                             curvatureAndPhi.curvature,
                             curvatureAndPhi.curvatureError,
                             curvatureAndPhi.phi);
    return vh;
  }

  return VectorHit();
}

void VectorHitBuilderAlgorithm::fit2Dzx(const Local3DPoint lpCI,
                                        const Local3DPoint lpCO,
                                        const LocalError leCI,
                                        const LocalError leCO,
                                        Local3DPoint& pos,
                                        Local3DVector& dir,
                                        AlgebraicSymMatrix22& covMatrix,
                                        double& chi2) const {
  float x[2] = {lpCI.z(), lpCO.z()};
  float y[2] = {lpCI.x(), lpCO.x()};
  float sqCI = sqrt(leCI.xx());
  float sqCO = sqrt(leCO.xx());
  float sigy2[2] = {sqCI * sqCI, sqCO * sqCO};

  fit(x, y, sigy2, pos, dir, covMatrix, chi2);

  return;
}

void VectorHitBuilderAlgorithm::fit2Dzy(const Local3DPoint lpCI,
                                        const Local3DPoint lpCO,
                                        const LocalError leCI,
                                        const LocalError leCO,
                                        Local3DPoint& pos,
                                        Local3DVector& dir,
                                        AlgebraicSymMatrix22& covMatrix,
                                        double& chi2) const {
  float x[2] = {lpCI.z(), lpCO.z()};
  float y[2] = {lpCI.y(), lpCO.y()};
  float sqCI = sqrt(leCI.yy());
  float sqCO = sqrt(leCO.yy());
  float sigy2[2] = {sqCI * sqCI, sqCO * sqCO};

  fit(x, y, sigy2, pos, dir, covMatrix, chi2);

  return;
}

void VectorHitBuilderAlgorithm::fit(float x[2],
                                    float y[2],
                                    float sigy2[2],
                                    Local3DPoint& pos,
                                    Local3DVector& dir,
                                    AlgebraicSymMatrix22& covMatrix,
                                    double& chi2) const {
  float slope = 0.;
  float intercept = 0.;
  float covss = 0.;
  float covii = 0.;
  float covsi = 0.;

  linearFit(x, y, 2, sigy2, slope, intercept, covss, covii, covsi);

  covMatrix[0][0] = covss;  // this is var(dy/dz)
  covMatrix[1][1] = covii;  // this is var(y)
  covMatrix[1][0] = covsi;  // this is cov(dy/dz,y)

  for (unsigned int j = 0; j < 2; j++) {
    const double ypred = intercept + slope * x[j];
    const double dy = (y[j] - ypred) / sqrt(sigy2[j]);
    chi2 += dy * dy;
  }

  pos = Local3DPoint(intercept, 0., 0.);
  //difference in z is the difference of the lowermost and the uppermost cluster z pos
  float slopeZ = x[1] - x[0];
  dir = LocalVector(slope, 0., slopeZ);
}

VectorHitBuilderAlgorithm::CurvatureAndPhi VectorHitBuilderAlgorithm::curvatureANDphi(Global3DPoint gPositionLower,
                                                                                      Global3DPoint gPositionUpper,
                                                                                      GlobalError gErrorLower,
                                                                                      GlobalError gErrorUpper) const {
  VectorHitBuilderAlgorithm::CurvatureAndPhi result;

  float curvature = -999.;
  float errorCurvature = -999.;
  float phi = -999.;

  float h1 = gPositionLower.x() * gPositionUpper.y() - gPositionUpper.x() * gPositionLower.y();

  //determine sign of curvature
  AlgebraicVector2 n1;
  n1[0] = -gPositionLower.y();
  n1[1] = gPositionLower.x();
  AlgebraicVector2 n2;
  n2[0] = gPositionUpper.x() - gPositionLower.x();
  n2[1] = gPositionUpper.y() - gPositionLower.y();

  double n3 = n1[0] * n2[0] + n1[1] * n2[1];
  double signCurv = -copysign(1.0, n3);
  double phi1 = atan2(gPositionUpper.y() - gPositionLower.y(), gPositionUpper.x() - gPositionLower.x());

  double x2Low = pow(gPositionLower.x(), 2);
  double y2Low = pow(gPositionLower.y(), 2);
  double x2Up = pow(gPositionUpper.x(), 2);
  double y2Up = pow(gPositionUpper.y(), 2);

  if (h1 != 0) {
    double h2 = 2 * h1;
    double h2Inf = 1. / (2 * h1);
    double r12 = gPositionLower.perp2();
    double r22 = gPositionUpper.perp2();
    double h3 = pow(n2[0], 2) + pow(n2[1], 2);
    double h4 = -x2Low * gPositionUpper.x() + gPositionLower.x() * x2Up + gPositionLower.x() * y2Up -
                gPositionUpper.x() * y2Low;
    double h5 =
        x2Low * gPositionUpper.y() - x2Up * gPositionLower.y() + y2Low * gPositionUpper.y() - gPositionLower.y() * y2Up;

    //radius of circle
    double invRho2 = (4. * h1 * h1) / (r12 * r22 * h3);
    curvature = sqrt(invRho2);

    //center of circle
    double xcentre = h5 / h2;
    double ycentre = h4 / h2;

    //to compute phi at the cluster points
    double xtg = gPositionLower.y() - ycentre;
    double ytg = -(gPositionLower.x() - xcentre);

    //to compute phi at the origin
    phi = atan2(ytg, xtg);

    AlgebraicROOTObject<4, 4>::Matrix jacobian;

    double denom1 = 1. / sqrt(r12 * r22 * h3);
    double denom2 = 1. / (pow(r12 * r22 * h3, 1.5));
    jacobian[0][0] = 1.0;  // dx1/dx1 dx1/dy1 dx2/dx1 dy2/dx1
    jacobian[1][1] = 1.0;  //dy1/dx1 dy1/dy1 dy2/dx1 dy2/dx1
    jacobian[2][0] =
        -2. * ((h1 * (gPositionLower.x() * r22 * h3 + (gPositionLower.x() - gPositionUpper.x()) * r12 * r22)) * denom2 -
               (gPositionUpper.y()) * denom1);  // dkappa/dx1
    jacobian[2][1] =
        -2. * ((gPositionUpper.x()) * denom1 +
               (h1 * (gPositionLower.y() * r22 * h3 + r12 * r22 * (gPositionLower.y() - gPositionUpper.y()))) *
                   denom2);  // dkappa/dy1
    jacobian[2][2] =
        -2. * ((gPositionLower.y()) * denom1 +
               (h1 * (gPositionUpper.x() * r12 * h3 - (gPositionLower.x() - gPositionUpper.x()) * r12 * r22)) *
                   denom2);  // dkappa/dx2
    jacobian[2][3] =
        -2. * ((h1 * (gPositionUpper.y() * r12 * h3 - r12 * r22 * (gPositionLower.y() - gPositionUpper.y()))) * denom2 -
               (gPositionLower.x()) * denom1);  // dkappa/dy2
    AlgebraicVector2 mVector;
    //to compute phi at the cluster points
    mVector[0] = (gPositionLower.y() - ycentre) * invRho2;   // dphi/dxcentre
    mVector[1] = -(gPositionLower.x() - xcentre) * invRho2;  // dphi/dycentre
    //to compute phi at the origin

    double h22Inv = 1. / pow(h2, 2);

    AlgebraicROOTObject<2, 4>::Matrix kMatrix;
    kMatrix[0][0] =
        2. * ((gPositionLower.x() * gPositionUpper.y()) * h2Inf - (gPositionUpper.y() * h5) * h22Inv);  // dxm/dx1
    kMatrix[0][1] = (2. * gPositionUpper.x() * h5) * h22Inv -
                    (x2Up + y2Up - 2. * gPositionLower.y() * gPositionUpper.y()) * h2Inf;  // dxm/dy1
    kMatrix[0][2] =
        2. * ((gPositionLower.y() * h5) * h22Inv - (gPositionUpper.x() * gPositionLower.y()) * h2Inf);  // dxm/dx2
    kMatrix[0][3] = (x2Low + y2Low - 2. * gPositionUpper.y() * gPositionLower.y()) * h2Inf -
                    (2. * gPositionLower.x() * h5) * h22Inv;  // dxm/dy2
    kMatrix[1][0] = (x2Up - 2. * gPositionLower.x() * gPositionUpper.x() + y2Up) * h2Inf -
                    (2. * gPositionUpper.y() * h4) * h22Inv;  // dym/dx1
    kMatrix[1][1] =
        2. * ((gPositionUpper.x() * h4) * h22Inv - (gPositionUpper.x() * gPositionLower.y()) * h2Inf);  // dym/dy1
    kMatrix[1][2] = (2. * gPositionLower.y() * h4) * h22Inv -
                    (x2Low - 2. * gPositionUpper.x() * gPositionLower.x() + y2Low) * h2Inf;  // dym/dx2
    kMatrix[1][3] =
        2. * (gPositionLower.x() * gPositionUpper.y()) * h2Inf - (gPositionLower.x() * h4) * h22Inv;  // dym/dy2

    AlgebraicVector4 nMatrix = mVector * kMatrix;
    jacobian[3][0] = nMatrix[0];  // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][1] = nMatrix[1];  // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][2] = nMatrix[2];  // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][3] = nMatrix[3];  // dphi/(dx1,dy1,dx2,dy2)

    //assign correct sign to the curvature errors
    if ((signCurv < 0 && curvature > 0) || (signCurv > 0 && curvature < 0)) {
      curvature = -curvature;
      for (int i = 0; i < 4; i++) {
        jacobian[2][i] = -jacobian[2][i];
      }
    }

    // bring phi in the same quadrant as phi1
    if (deltaPhi(phi, phi1) > M_PI / 2.) {
      phi = phi + M_PI;
      if (phi > M_PI)
        phi = phi - 2. * M_PI;
    }

    //computing the curvature error
    AlgebraicVector4 curvatureJacobian;
    for (int i = 0; i < 4; i++) {
      curvatureJacobian[i] = jacobian[2][i];
    }

    AlgebraicROOTObject<4, 4>::Matrix gErrors;

    gErrors[0][0] = gErrorLower.cxx();
    gErrors[0][1] = gErrorLower.cyx();
    gErrors[1][0] = gErrorLower.cyx();
    gErrors[1][1] = gErrorLower.cyy();
    gErrors[2][2] = gErrorUpper.cxx();
    gErrors[2][3] = gErrorUpper.cyx();
    gErrors[3][2] = gErrorUpper.cyx();
    gErrors[3][3] = gErrorUpper.cyy();

    AlgebraicVector4 temp = curvatureJacobian;
    temp = temp * gErrors;
    errorCurvature = temp[0] * curvatureJacobian[0] + temp[1] * curvatureJacobian[1] + temp[2] * curvatureJacobian[2] +
                     temp[3] * curvatureJacobian[3];
  }

  result.curvature = curvature;
  result.curvatureError = errorCurvature;
  result.phi = phi;
  return result;
}
