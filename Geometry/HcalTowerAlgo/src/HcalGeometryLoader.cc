#include "Geometry/HcalTowerAlgo/interface/HcalGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include <iostream>

namespace cms {

HcalGeometryLoader::HcalGeometryLoader() 
{
  theBarrelDepth = 190.;
  theHBThickness = 93.6; // just from drawings.  All thicknesses needs to be done right
  theOuterDepth  = 406;
  theHOThickness = 1.;

  theHEDepth[0] = 388.0;
  theHEDepth[1] = 397.0;
  theHEDepth[2] = 450.0;
  theHEDepth[3] = 568.0;

  theHFDepth[0] = 1100.0;
  theHFDepth[1] = 1120.0;
  theHFThickness = 165;

  // the first tower all have width = 0.087
  for(unsigned i = 0; i < 20; ++i) {
    theHBHEEtaBounds[i] = i* 0.087;
  }
  theHBHEEtaBounds[20] = 1.74;
  theHBHEEtaBounds[21] = 1.83;
  theHBHEEtaBounds[22] = 1.93;
  theHBHEEtaBounds[23] = 2.043;
  theHBHEEtaBounds[24] = 2.172;
  theHBHEEtaBounds[25] = 2.322;
  theHBHEEtaBounds[26] = 2.5;
  theHBHEEtaBounds[27] = 2.65;
  theHBHEEtaBounds[28] = 2.868;
  theHBHEEtaBounds[29] = 3.;

  theHFEtaBounds[ 0] = 2.853;
  theHFEtaBounds[ 1] = 2.964;
  theHFEtaBounds[ 2] = 3.139;
  theHFEtaBounds[ 3] = 3.314;
  theHFEtaBounds[ 4] = 3.489;
  theHFEtaBounds[ 5] = 3.664;
  theHFEtaBounds[ 6] = 3.839;
  theHFEtaBounds[ 7] = 4.013;
  theHFEtaBounds[ 8] = 4.191;
  theHFEtaBounds[ 9] = 4.363;
  theHFEtaBounds[10] = 4.538;
  theHFEtaBounds[11] = 4.716;
  theHFEtaBounds[12] = 4.889;
  theHFEtaBounds[13] = 5.191;

}
void HcalGeometryLoader::fill(std::vector<DetId> & detIds, std::map<DetId, const CaloCellGeometry*> geometries) {

  fill(HcalBarrel, theTopology.firstHBRing(), theTopology.lastHBRing(), detIds, geometries);

  fill(HcalEndcap, theTopology.firstHERing(), theTopology.lastHERing(), detIds, geometries);

  fill(HcalForward, theTopology.firstHFRing(), theTopology.lastHFRing(), detIds, geometries);

  fill(HcalOuter, theTopology.firstHORing(), theTopology.lastHORing(), detIds, geometries);
}


void HcalGeometryLoader::fill(HcalSubdetector subdet, int firstEtaRing, int lastEtaRing, 
                             std::vector<DetId> & detIds, std::map<DetId, const CaloCellGeometry*> geometries) 
{
  // start by making the new HcalDetIds
 std::vector<HcalDetId> hcalIds;
  int nDepthSegments, startingDepthSegment;
  for(int etaRing = firstEtaRing; etaRing <= lastEtaRing; ++etaRing) {
    theTopology.depthBinInformation(subdet, etaRing, nDepthSegments, startingDepthSegment);
    int nPhiSegments = theTopology.nPhiBins(etaRing);
    for(int idepth = 0; idepth < nDepthSegments; ++idepth) {
      int depthBin = startingDepthSegment + idepth;

      for(unsigned iphi = 1; iphi <= nPhiSegments; ++iphi) {
        for(int zsign = -1; zsign <= 1; zsign += 2) {
           hcalIds.push_back(HcalDetId( subdet, zsign * etaRing, iphi, depthBin) );
        }
      } 
    }
  }

std::cout << "Number of HCAL DetIds made: " << subdet << " " << hcalIds.size() << std::endl;
  // for each new HcalDetId, make a CaloCellGeometry
  for(std::vector<HcalDetId>::const_iterator hcalIdItr = hcalIds.begin();
      hcalIdItr != hcalIds.end(); ++hcalIdItr)
  {
    DetId detId(*hcalIdItr);
    detIds.push_back(detId);
    CaloCellGeometry * geometry = makeCell(*hcalIdItr);
    geometries.insert(std::pair<DetId, CaloCellGeometry*>(detId, geometry));
  }
}
     

inline double theta_from_eta(double eta){return (2.0*atan(exp(-eta)));}


CaloCellGeometry * HcalGeometryLoader::makeCell(const HcalDetId & detId) const {

  // the two eta boundaries of the cell
  double eta1, eta2;
  HcalSubdetector subdet = detId.subdet();
  int etaRing = detId.ietaAbs();
  if(subdet == HcalForward) {
    eta1 = theHFEtaBounds[etaRing-theTopology.firstHFRing()];
    eta2 = theHFEtaBounds[etaRing-theTopology.firstHFRing()+1];
  } else {
    eta1 = theHBHEEtaBounds[etaRing-1];
    eta2 = theHBHEEtaBounds[etaRing];
  }
  double eta = 0.5*(eta1+eta2) * detId.zside();
  double deta = 0.5*(eta2-eta1);
  double theta = theta_from_eta(eta);

  // is this right?  there's a cell centered at phi=0?
  double startingPhi = 0.;
  // in radians
  double dphi = M_PI / theTopology.nPhiBins(etaRing);
  double phi = startingPhi + dphi*(detId.iphi()-1);

  bool isBarrel = (subdet == HcalBarrel || subdet == HcalOuter);

  double x,y,z,r;
  double thickness;

  if(isBarrel) {
    if(subdet == HcalBarrel) {
      r = theBarrelDepth;
      thickness = theHBThickness;
    } else { // HO
      r = theOuterDepth;
      thickness = theHOThickness;
    } 
    z = r * cos(theta);

  } else {

    int depth = detId.depth();
    if(subdet == HcalEndcap) {
      z = theHEDepth[depth - 1];
      thickness = theHEDepth[depth] - z;
    } else {
      z = theHFDepth[depth - 1];
      thickness = theHFThickness;
    }
    r = z * sin(theta);
    assert(r>0.);

  }

  x = r * cos(phi);
  y = r * sin(phi);
  GlobalPoint point(x,y,z);

  return new calogeom::IdealObliquePrism(point, deta, dphi, thickness, !isBarrel);

}



}

