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
typedef ROOT::Math::Transform3DPJ::Point Point;

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
  totalLayer1_(0.),totalLayer2_(0.),
  theGenerator(aGenerator)
{
  double dummyt;
  anglecorrection1_ = 0.;
  anglecorrection2_ = 0.;
   // Check if the entrance points are really on the wafers
  // Layer 1 
  layer1valid_ = (layer1entrance.Mag2()>0.);
  if(layer1valid_)
    {
      int z=(psLayer1Entrance_.z()>0)? 1:-1;
      Plane3D plan1(0.,0.,1.,-z*myCalorimeter->preshowerZPosition(1));

      psLayer1Entrance_ = intersect(plan1,layer1entrance,layer1entrance+layer1dir,dummyt,false);

      XYZVector zaxis(0,0,1);
      XYZVector planeVec1=(zaxis.Cross(layer1dir)).Unit();
      locToGlobal1_=Transform3D(Point(0,0,0),
				Point(0,0,1),
				Point(1,0,0),
				(Point)psLayer1Entrance_,
				(Point)(psLayer1Entrance_+layer1dir),
				(Point)(psLayer1Entrance_+planeVec1));

      anglecorrection1_ = fabs(zaxis.Dot(layer1dir));
      if(anglecorrection1_!=0.) anglecorrection1_ = 1./anglecorrection1_;
      //      std::cout << " Layer 1 entrance " << psLayer1Entrance_ << std::endl;
      //      std::cout << " Layer 1 corr " << anglecorrection1_ << std::endl;
    }

  // Layer 2
  layer2valid_ = (layer2entrance.Mag2()>0.);
  if(layer2valid_)
    {
      int z=(psLayer2Entrance_.z()>0) ? 1:-1;
      Plane3D plan2(0.,0.,1.,-z*myCalorimeter->preshowerZPosition(2));
      
      psLayer2Entrance_ = intersect(plan2,layer2entrance,layer2entrance+layer2dir,dummyt,false);

      XYZVector zaxis(0,0,1);
      XYZVector planeVec2=(zaxis.Cross(layer2dir)).Unit();
      locToGlobal2_=Transform3D(Point(0,0,0),
				Point(0,0,1),
				Point(1,0,0),
			       (Point)psLayer2Entrance_,
			       (Point)(psLayer2Entrance_+layer2dir),
			       (Point)(psLayer2Entrance_+planeVec2));
      
      anglecorrection2_ = fabs(zaxis.Dot(layer2dir));
      if(anglecorrection2_!=0.) anglecorrection2_ = 1./anglecorrection2_;
      //      std::cout << " Layer 2 entrance " << psLayer2Entrance_ << std::endl;
      //      std::cout << " Layer 2 corr " << anglecorrection2_ << std::endl;
    }
  //  theGenerator=LandauFluctuationGenerator();
}


bool
PreshowerHitMaker::addHit(double r,double phi,unsigned layer)
{
  if((layer==1&&!layer1valid_)||((layer==2&&!layer2valid_))) return false;

  r*=moliereRadius;
  XYZPoint point (r*std::cos(phi),r*std::sin(phi),0.);
  point =  (layer==1) ? locToGlobal1_((Point)point) : locToGlobal2_((Point)point);
  //  std::cout << "  Point " << point  << std::endl;
  int z=(point.z()>0) ? 1: -1;
  point = XYZPoint(point.x(),point.y(),z*myCalorimeter->preshowerZPosition(layer));
  //  std::cout << "r " << r << "  Point after " << point  << std::endl;
  //  std::cout << " Layer " << layer << " " << point << std::endl;
  DetId strip = myCalorimeter->getEcalPreshowerGeometry()->getClosestCellInPlane(GlobalPoint(point.x(),point.y(),point.z()),layer);

  float meanspot=(layer==1) ? mip1_ : mip2_; 
  float spote = meanspot + 0.000021*theGenerator->landau();
  spote *= ( (layer==1) ? anglecorrection1_ : anglecorrection2_ );

  if(!strip.null())
    {
      //calculate time of flight
      double tof = (myCalorimeter->getEcalPreshowerGeometry()->getGeometry(strip)->getPosition().mag())/29.98;//speed of light
	  CaloHitID current_id(strip.rawId(),tof,0); //no track yet
      std::map<CaloHitID,float>::iterator cellitr;
      cellitr = hitMap_.find(current_id);
      if( cellitr==hitMap_.end())
	{
	  hitMap_.insert(std::pair<CaloHitID,float>(current_id,spote));
	}
      else
	{
	  cellitr->second+=spote;
	}  
      //      std::cout << " found " << stripNumber << " " << spote <<std::endl;
      if(layer==1){
	totalLayer1_+=spote;
      }
      else if (layer==2) {
	totalLayer2_+=spote;
      }
      return true;
    }
  //  std::cout << "  Could not find a cell " << point << std::endl;
  return false;
}
