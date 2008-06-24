#include "FastSimulation/CaloHitMakers/interface/PreshowerHitMaker.h"
#
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/Utilities/interface/LandauFluctuationGenerator.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

#include "Math/GenVector/Plane3D.h"

#include <cmath>

typedef ROOT::Math::Plane3D Plane3D;

// LandauFluctuationGenerator PreshowerHitMaker::theGenerator=LandauFluctuationGenerator();

PreshowerHitMaker::PreshowerHitMaker(
    CaloGeometryHelper * calo,
    const XYZPoint& layer1entrance, 
    const XYZVector& layer1dir, 
    const XYZPoint& layer2entrance, 
    const XYZVector& layer2dir,
    const LandauFluctuationGenerator* aGenerator):
  CaloHitMaker(calo,DetId::Ecal,EcalPreshower,2),
  psLayer1Entrance_(layer1entrance),
  psLayer1Dir_(layer1dir),
  psLayer2Entrance_(layer2entrance),
  psLayer2Dir_(layer2dir),
  theGenerator(aGenerator)
{
   // Check if the entrance points are really on the wafers
  // Layer 1 
  int z=(psLayer1Entrance_.z()>0)? 1:-1;
  Plane3D plan1(0.,0.,1.,-z*myCalorimeter->preshowerZPosition(1));
  double dummyt;
  psLayer1Entrance_ = intersect(plan1,layer1entrance,layer1entrance+layer1dir,dummyt,false);
  x1=psLayer1Entrance_.x();
  y1=psLayer1Entrance_.y();
  z1=psLayer1Entrance_.z();
  XYZVector dirx(psLayer1Entrance_.x(),0,psLayer1Entrance_.z());
  XYZVector diry(0,psLayer1Entrance_.y(),psLayer1Entrance_.z());
  dirx=dirx.Unit();
  diry=diry.Unit();
  invcostheta1x = 1./fabs(dirx.Dot(XYZVector(0,0,1.)));
  invcostheta1y = 1./fabs(diry.Dot(XYZVector(0,0,1.)));      

  // Layer 2
  z=(psLayer2Entrance_.z()>0) ? 1:-1;
  Plane3D plan2(0.,0.,1.,-z*myCalorimeter->preshowerZPosition(2));
  
  psLayer2Entrance_ = intersect(plan2,layer2entrance,layer2entrance+layer2dir,dummyt,false);
  x2=psLayer2Entrance_.x();
  y2=psLayer2Entrance_.y();
  z2=psLayer2Entrance_.z();
  dirx = XYZVector(psLayer2Entrance_.x(),0,psLayer2Entrance_.z());
  diry = XYZVector(0,psLayer2Entrance_.y(),psLayer2Entrance_.z());
  dirx=dirx.Unit();
  diry=diry.Unit();
  invcostheta2x = 1./fabs(dirx.Dot(XYZVector(0,0,1.)));
  invcostheta2y = 1./fabs(diry.Dot(XYZVector(0,0,1.)));

  //  theGenerator=LandauFluctuationGenerator();
}


bool 
PreshowerHitMaker::addHit(double r,double phi,unsigned layer)
{
  r*=moliereRadius;
  XYZPoint point = (layer==1) ? 
    XYZPoint(x1+r*invcostheta1x*std::cos(phi),y1+r*invcostheta1y*std::sin(phi),z1) : 
    XYZPoint(x2+r*invcostheta2x*std::cos(phi),y2+r*invcostheta2y*std::sin(phi),z2);
  
  //  std::cout << " Layer " << layer << " " << point << std::endl;
  DetId strip = myCalorimeter->getEcalPreshowerGeometry()->getClosestCellInPlane(GlobalPoint(point.x(),point.y(),point.z()),layer);

  float spote=0.000095+0.000021*theGenerator->landau();

  if(!strip.null())
    {
      uint stripNumber=strip.rawId();
      std::map<uint32_t,float>::iterator cellitr;
      cellitr = hitMap_.find(stripNumber);
      if( cellitr==hitMap_.end())
	{
	  hitMap_.insert(std::pair<uint32_t,float>(stripNumber,spote));
	}
      else
	{
	  cellitr->second+=spotEnergy;
	}  
      return true;
    }
  return false;
}
