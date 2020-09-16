#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithm.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit2D.h"

void VectorHitBuilderAlgorithm::run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                                    VectorHitCollectionNew& vhAcc,
                                    VectorHitCollectionNew& vhRej,
                                    edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
                                    edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej) const {
  LogDebug("VectorHitBuilderAlgorithm") << "Run VectorHitBuilderAlgorithm ... \n";
  const edmNew::DetSetVector<Phase2TrackerCluster1D>* ClustersPhase2Collection = clusters.product();

  std::unordered_map<DetId, std::vector<VectorHit>> tempVHAcc, tempVHRej;

  //loop over the DetSetVector
  LogDebug("VectorHitBuilderAlgorithm") << "with #clusters : " << ClustersPhase2Collection->size() << std::endl;
  for (auto DSViter : *ClustersPhase2Collection) {
    unsigned int rawDetId1(DSViter.detId());
    DetId detId1(rawDetId1);
    DetId lowerDetId, upperDetId;
    if (theTkTopo->isLower(detId1)) {
      lowerDetId = detId1;
      upperDetId = theTkTopo->partnerDetId(detId1);
    } else if (theTkTopo->isUpper(detId1)) {
      continue;
    }
    DetId detIdStack = theTkTopo->stack(detId1);

    //debug
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId stack : " << detIdStack.rawId() << std::endl;
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId lower set of clusters  : " << lowerDetId.rawId();
    LogDebug("VectorHitBuilderAlgorithm") << "  DetId upper set of clusters  : " << upperDetId.rawId() << std::endl;

    const GeomDet* gd;
    const StackGeomDet* stackDet;
    const auto& it_detLower = ClustersPhase2Collection->find(lowerDetId);
    const auto& it_detUpper = ClustersPhase2Collection->find(upperDetId);

    if (it_detLower != ClustersPhase2Collection->end() && it_detUpper != ClustersPhase2Collection->end()) {
      gd = theTkGeom->idToDet(detIdStack);
      stackDet = dynamic_cast<const StackGeomDet*>(gd);
      std::vector<VectorHit> vhsInStack_Acc;
      std::vector<VectorHit> vhsInStack_Rej;
      const auto& vhsInStack_AccRej = buildVectorHits(stackDet, clusters, *it_detLower, *it_detUpper);

      //storing accepted and rejected VHs
      for (const auto& vh : vhsInStack_AccRej) {
        if (vh.second == true) {
          vhsInStack_Acc.push_back(vh.first);
          std::push_heap(vhsInStack_Acc.begin(),vhsInStack_Acc.end());
        } else if (vh.second == false) {
          vhsInStack_Rej.push_back(vh.first);
        }
      }

      //ERICA:: to be checked with map!
      //sorting vhs for best chi2
      std::sort_heap(vhsInStack_Acc.begin(), vhsInStack_Acc.end());

      tempVHAcc[detIdStack] = vhsInStack_Acc;
      tempVHRej[detIdStack] = vhsInStack_Rej;
#ifdef EDM_ML_DEBUG
      LogTrace("VectorHitBuilderAlgorithm")
          << "For detId #" << detIdStack.rawId() << " the following VHits have been accepted:";
      for (const auto& vhIt : vhsInStack_Acc) {
        LogTrace("VectorHitBuilderAlgorithm") << "accepted VH: " << vhIt;
      }
      LogTrace("VectorHitBuilderAlgorithm")
          << "For detId #" << detIdStack.rawId() << " the following VHits have been rejected:";
      for (const auto& vhIt : vhsInStack_Rej) {
        LogTrace("VectorHitBuilderAlgorithm") << "rejected VH: " << vhIt;
      }
#endif
    }
  }

  loadDetSetVector(tempVHAcc, vhAcc);
  loadDetSetVector(tempVHRej, vhRej);

  LogDebug("VectorHitBuilderAlgorithm") << "End run VectorHitBuilderAlgorithm ... \n";
  return;
}

bool VectorHitBuilderAlgorithm::checkClustersCompatibilityBeforeBuilding(
    edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
    const detset& theLowerDetSet,
    const detset& theUpperDetSet) const {
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
std::vector<std::pair<VectorHit, bool>> VectorHitBuilderAlgorithm::buildVectorHits(
    const StackGeomDet* stack,
    edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
    const detset& theLowerDetSet,
    const detset& theUpperDetSet,
    const std::vector<bool>& phase2OTClustersToSkip) const {
  std::vector<std::pair<VectorHit, bool>> result;
  if (checkClustersCompatibilityBeforeBuilding(clusters, theLowerDetSet, theUpperDetSet)) {
    LogDebug("VectorHitBuilderAlgorithm") << "  compatible -> continue ... " << std::endl;
  } else {
    LogTrace("VectorHitBuilderAlgorithm") << "  not compatible, going to the next cluster";
  }
  //only cache local parameters for upper cluster as we loop over lower clusters only once anyway
  std::vector<std::pair<LocalPoint, LocalError>> localParamsUpper;
  std::vector<const PixelGeomDetUnit*> localGDUUpper;

  std::vector<Phase2TrackerCluster1DRef> upperClusters;
  for (const_iterator ciu = theUpperDetSet.begin(); ciu != theUpperDetSet.end(); ++ciu) {
    Phase2TrackerCluster1DRef clusterUpper = edmNew::makeRefTo(clusters, ciu);
    const PixelGeomDetUnit* gduUpp = dynamic_cast<const PixelGeomDetUnit*>(stack->upperDet());
    localGDUUpper.push_back(gduUpp);
    localParamsUpper.push_back(theCpe->localParameters(*clusterUpper, *gduUpp));
  }
  int upperIterator = 0;
  for (const_iterator cil = theUpperDetSet.begin(); cil != theUpperDetSet.end(); ++cil) {
    LogDebug("VectorHitBuilderAlgorithm") << " lower clusters " << std::endl;
    Phase2TrackerCluster1DRef cluL = edmNew::makeRefTo(clusters, cil);
#ifdef EDM_ML_DEBUG
    printCluster(stack->lowerDet(), &*cluL);
#endif
    const PixelGeomDetUnit* gduLow = dynamic_cast<const PixelGeomDetUnit*>(stack->lowerDet());
    auto&& lparamsLow = theCpe->localParameters(*cluL, *gduLow);
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
      if (localParamsUpper[upperIterator].first.x() > lparamsLow.first.x()) {
        if (localParamsUpper[upperIterator].first.x() > 0) {
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = localParamsUpper[upperIterator].first.x() - std::abs(pC);
        } else if (localParamsUpper[upperIterator].first.x() < 0) {
          lpos_low_corr = lparamsLow.first.x() + std::abs(pC);
          lpos_upp_corr = localParamsUpper[upperIterator].first.x();
        }
      } else if (localParamsUpper[upperIterator].first.x() < lparamsLow.first.x()) {
        if (localParamsUpper[upperIterator].first.x() > 0) {
          lpos_low_corr = lparamsLow.first.x() - std::abs(pC);
          lpos_upp_corr = localParamsUpper[upperIterator].first.x();
        } else if (localParamsUpper[upperIterator].first.x() < 0) {
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = localParamsUpper[upperIterator].first.x() + std::abs(pC);
        }
      } else {
        if (localParamsUpper[upperIterator].first.x() > 0) {
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = localParamsUpper[upperIterator].first.x() - std::abs(pC);
        } else if (localParamsUpper[upperIterator].first.x() < 0) {
          lpos_low_corr = lparamsLow.first.x();
          lpos_upp_corr = localParamsUpper[upperIterator].first.x() + std::abs(pC);
        }
      }

      LogDebug("VectorHitBuilderAlgorithm") << " \t local pos upper corrected (x):" << lpos_upp_corr << std::endl;
      LogDebug("VectorHitBuilderAlgorithm") << " \t local pos lower corrected (x):" << lpos_low_corr << std::endl;

      //building my tolerance : 10*sigma
      double delta = 10.0 * sqrt(lparamsLow.second.xx() + localParamsUpper[upperIterator].second.xx());
      LogDebug("VectorHitBuilderAlgorithm") << " \t delta: " << delta << std::endl;

      double width = lpos_low_corr - lpos_upp_corr;
      LogDebug("VectorHitBuilderAlgorithm") << " \t width: " << width << std::endl;

      unsigned int layerStack = theTkTopo->layer(stack->geographicalId());
      if (stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTB)
        LogDebug("VectorHitBuilderAlgorithm") << " \t is barrel.    " << std::endl;
      if (stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC)
        LogDebug("VectorHitBuilderAlgorithm") << " \t is endcap.    " << std::endl;
      LogDebug("VectorHitBuilderAlgorithm") << " \t layer is : " << layerStack << std::endl;

      float cut = 0.0;
      if (stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTB)
        cut = barrelCut.at(layerStack);
      if (stack->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC)
        cut = endcapCut.at(layerStack);
      LogDebug("VectorHitBuilderAlgorithm") << " \t the cut is:" << cut << std::endl;

      //old cut: indipendent from layer
      //if( (lpos_upp_corr < lpos_low_corr + delta) &&
      //    (lpos_upp_corr > lpos_low_corr - delta) ){
      //new cut: dependent on layers
      if (std::abs(width) < cut) {
        LogDebug("VectorHitBuilderAlgorithm") << " accepting VH! " << std::endl;
        VectorHit vh = buildVectorHit(stack, cluL, cluU);
        //protection: the VH can also be empty!!
        if (vh.isValid()) {
          result.emplace_back(std::make_pair(vh, true));
        }

      } else {
        LogDebug("VectorHitBuilderAlgorithm") << " rejecting VH: " << std::endl;
        //storing vh rejected for combinatiorial studies
        VectorHit vh = buildVectorHit(stack, cluL, cluU);
        result.emplace_back(std::make_pair(vh, false));
      }
      upperIterator = +1;
    }
  }

  return result;
}

VectorHit VectorHitBuilderAlgorithm::buildVectorHit(const StackGeomDet* stack,
                                                    Phase2TrackerCluster1DRef lower,
                                                    Phase2TrackerCluster1DRef upper) const {
  LogTrace("VectorHitBuilderAlgorithm") << "Build VH with: ";
#ifdef EDM_ML_DEBUG
  printCluster(stack->upperDet(),&*upper);
#endif
  const PixelGeomDetUnit* geomDetLower = static_cast<const PixelGeomDetUnit*>(stack->lowerDet());
  const PixelGeomDetUnit* geomDetUpper = static_cast<const PixelGeomDetUnit*>(stack->upperDet());

  auto&& lparamsLower = theCpe->localParameters(*lower, *geomDetLower);  // x, y, z, e2_xx, e2_xy, e2_yy
  Global3DPoint gparamsLower = geomDetLower->surface().toGlobal(lparamsLower.first);
  LogTrace("VectorHitBuilderAlgorithm") << "\t lower global pos: " << gparamsLower;

  auto&& lparamsUpper = theCpe->localParameters(*upper, *geomDetUpper);
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
      gPositionLower = VectorHit::phase2clusterGlobalPos(geomDetUpper, upper);
      gPositionUpper = VectorHit::phase2clusterGlobalPos(geomDetLower, lower);
      gErrorLower = VectorHit::phase2clusterGlobalPosErr(geomDetUpper);
      gErrorUpper = VectorHit::phase2clusterGlobalPosErr(geomDetLower);
    }

    const auto& curvatureAndPhi = curvatureANDphi(gPositionLower, gPositionUpper, gErrorLower, gErrorUpper);
    VectorHit vh = VectorHit(*stack, vh2Dzx, vh2Dzy, lowerOmni, upperOmni, curvatureAndPhi.first.first, curvatureAndPhi.first.second, curvatureAndPhi.second);
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
  float sigy[2] = {sqCI, sqCO};

  fit(x, y, sigy, pos, dir, covMatrix, chi2);

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
  float sigy[2] = {sqCI, sqCO};

  fit(x, y, sigy, pos, dir, covMatrix, chi2);

  return;
}

void VectorHitBuilderAlgorithm::fit(float x[2],
                                    float y[2],
                                    float sigy[2],
                                    Local3DPoint& pos,
                                    Local3DVector& dir,
                                    AlgebraicSymMatrix22& covMatrix,
                                    double& chi2) const {
  float slope = 0.;
  float intercept = 0.;
  float covss = 0.;
  float covii = 0.;
  float covsi = 0.;

  //theFitter->linearFit(x, y, 2, sigy, slope, intercept, covss, covii, covsi);
  linearFit(x, y, 2, sigy, slope, intercept, covss, covii, covsi);

  covMatrix[0][0] = covss;  // this is var(dy/dz)
  covMatrix[1][1] = covii;  // this is var(y)
  covMatrix[1][0] = covsi;  // this is cov(dy/dz,y)

  for (unsigned int j = 0; j < 2; j++) {
    const double ypred = intercept + slope * x[j];
    const double dy = (y[j] - ypred) / sigy[j];
    chi2 += dy * dy;
  }

  pos = Local3DPoint(intercept, 0., 0.);
  //difference in z is the difference of the lowermost and the uppermost cluster z pos
  float slopeZ = x[1] - x[0];
  dir = LocalVector(slope, 0., slopeZ);
}

std::pair<std::pair<float, float>,float> VectorHitBuilderAlgorithm::curvatureANDphi(Global3DPoint gPositionLower,
                                                                  Global3DPoint gPositionUpper,
                                                                  GlobalError gErrorLower,
                                                                  GlobalError gErrorUpper) const {
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

  if (h1 != 0) {
    double h2 = 2 * h1;
    double h2Inf = 1. / (2 * h1);
    double r12 = pow(gPositionLower.x(), 2) + pow(gPositionLower.y(), 2);
    double r22 = pow(gPositionUpper.x(), 2) + pow(gPositionUpper.y(), 2);
    double h3 =
        (pow(gPositionLower.x(), 2) - 2. * gPositionLower.x() * gPositionUpper.x() + pow(gPositionUpper.x(), 2) +
         pow(gPositionLower.y(), 2) - 2. * gPositionLower.y() * gPositionUpper.y() + pow(gPositionUpper.y(), 2));
    double h4 = -pow(gPositionLower.x(), 2) * gPositionUpper.x() + gPositionLower.x() * pow(gPositionUpper.x(), 2) +
                gPositionLower.x() * pow(gPositionUpper.y(), 2) - gPositionUpper.x() * pow(gPositionLower.y(), 2);
    double h5 = pow(gPositionLower.x(), 2) * gPositionUpper.y() - pow(gPositionUpper.x(), 2) * gPositionLower.y() +
                pow(gPositionLower.y(), 2) * gPositionUpper.y() - gPositionLower.y() * pow(gPositionUpper.y(), 2);

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
    AlgebraicVector2 M;
    //to compute phi at the cluster points
    M[0] = (gPositionLower.y() - ycentre) * invRho2;   // dphi/dxcentre
    M[1] = -(gPositionLower.x() - xcentre) * invRho2;  // dphi/dycentre
    //to compute phi at the origin

    AlgebraicROOTObject<2, 4>::Matrix K;
    K[0][0] =
        2. * ((gPositionLower.x() * gPositionUpper.y()) * h2Inf - (gPositionUpper.y() * h5) / pow(h2, 2));  // dxm/dx1
    K[0][1] = (2. * gPositionUpper.x() * h5) / pow(h2, 2) -
              (pow(gPositionUpper.x(), 2) + pow(gPositionUpper.y(), 2) - 2. * gPositionLower.y() * gPositionUpper.y()) *
                  h2Inf;  // dxm/dy1
    K[0][2] =
        2. * ((gPositionLower.y() * h5) / pow(h2, 2) - (gPositionUpper.x() * gPositionLower.y()) * h2Inf);  // dxm/dx2
    K[0][3] = (pow(gPositionLower.x(), 2) + pow(gPositionLower.y(), 2) - 2. * gPositionUpper.y() * gPositionLower.y()) *
                  h2Inf -
              (2. * gPositionLower.x() * h5) / pow(h2, 2);  // dxm/dy2
    K[1][0] = (pow(gPositionUpper.x(), 2) - 2. * gPositionLower.x() * gPositionUpper.x() + pow(gPositionUpper.y(), 2)) *
                  h2Inf -
              (2. * gPositionUpper.y() * h4) / pow(h2, 2);  // dym/dx1
    K[1][1] =
        2. * ((gPositionUpper.x() * h4) / pow(h2, 2) - (gPositionUpper.x() * gPositionLower.y()) * h2Inf);  // dym/dy1
    K[1][2] = (2. * gPositionLower.y() * h4) / pow(h2, 2) -
              (pow(gPositionLower.x(), 2) - 2. * gPositionUpper.x() * gPositionLower.x() + pow(gPositionLower.y(), 2)) *
                  h2Inf;  // dym/dx2
    K[1][3] =
        2. * (gPositionLower.x() * gPositionUpper.y()) * h2Inf - (gPositionLower.x() * h4) / pow(h2, 2);  // dym/dy2

    AlgebraicVector4 N = M * K;
    jacobian[3][0] = N[0];  // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][1] = N[1];  // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][2] = N[2];  // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][3] = N[3];  // dphi/(dx1,dy1,dx2,dy2)

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

  } else {
    return std::make_pair(std::make_pair(0.,0.), 0.0);
  }
/*  switch (curvORphi) {
    case curvatureMode:
      return std::make_pair(curvature, errorCurvature);
    case phiMode:
      return std::make_pair(phi, 0.0);
  }*/
  return std::make_pair(std::make_pair(curvature, errorCurvature), phi);
}
