#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"

#include <algorithm>
#include <cmath>

typedef ROOT::Math::Transform3DPJ::Point Point;

HcalHitMaker::HcalHitMaker(EcalHitMaker& grid,unsigned shower)
  :CaloHitMaker(grid.getCalorimeter(),DetId::Hcal,HcalHitMaker::getSubHcalDet(grid.getFSimTrack()),
		grid.getFSimTrack()->onHcal()?grid.getFSimTrack()->onHcal():grid.getFSimTrack()->onVFcal()+1,shower),
   myGrid(grid),  myTrack((grid.getFSimTrack()))
{
  // normalize the direction
  ecalEntrance_=myGrid.ecalEntrance();
  particleDirection=myTrack->ecalEntrance().Vect().Unit();
  radiusFactor_=(EMSHOWER)? moliereRadius:interactionLength;
  mapCalculated_=false;
  //std::cout << " Famos HCAL " << grid.getTrack()->onHcal() << " " <<  grid.getTrack()->onVFcal() << " " << showerType << std::endl;
  if(EMSHOWER&&(abs(grid.getFSimTrack()->type())!=11 && grid.getFSimTrack()->type()!=22))
    {
      std::cout << " FamosHcalHitMaker : Strange. The following shower has EM type" << std::endl <<* grid.getFSimTrack() << std::endl;
    }
}

bool 
HcalHitMaker::addHit(double r,double phi,unsigned layer)
{
    //  std::cout << " FamosHcalHitMaker::addHit - radiusFactor = " << radiusFactor
  //	    << std::endl;

  XYZPoint point(r*radiusFactor_*std::cos(phi),r*radiusFactor_*std::sin(phi),0.);

  //  std::cout << " FamosHcalHitMaker::addHit - point before " << point << std::endl;

  point = locToGlobal_((Point)point);
  point = locToGlobal_ * point;

  //  std::cout << " FamosHcalHitMaker::addHit - point after  " << point << std::endl;

  // Temporary nasty hack to avoid misbehaviour of not-intended-for-that
  //  getClosestCell in case of large (eta beyond HF ...) 
  double pointeta = fabs(point.eta());
  if(pointeta > 5.19) return false; 

  DetId thecellID(myCalorimeter->getClosestCell(point,false,false));

  if(!thecellID.null())
    {

      uint32_t cell(thecellID.rawId());
  
      //      std::cout << " FamosHcalHitMaker::addHit - the cell num " << cell
      //      		<< std::endl;

      std::map<uint32_t,float>::iterator cellitr;
      cellitr = hitMap_.find(cell);
      if(cellitr==hitMap_.end())
	{
	  hitMap_.insert(std::pair<uint32_t,float>(cell,spotEnergy));
	}
      else
	{
	  cellitr->second+=spotEnergy;
	}
      return true;
    }
  return false;
}

bool 
HcalHitMaker::setDepth(double depth)
{
  currentDepth_=depth;
  std::vector<CaloSegment>::const_iterator segiterator;
  if(EMSHOWER)
    segiterator = find_if(myGrid.getSegments().begin(),myGrid.getSegments().end(),CaloSegment::inX0Segment(currentDepth_));
  
  //Hadron shower 
  if(HADSHOWER)
    segiterator = find_if(myGrid.getSegments().begin(),myGrid.getSegments().end(),CaloSegment::inL0Segment(currentDepth_));
  if(segiterator==myGrid.getSegments().end()) 
    {
      std::cout << " FamosHcalHitMaker : Could not go at such depth " << depth << std::endl;
      std::cout << " EMSHOWER " << EMSHOWER << std::endl;
      std::cout << " Track " << *(myGrid.getFSimTrack()) << std::endl;
      return false;
    }
  XYZPoint origin;
  if(EMSHOWER)
    origin=segiterator->positionAtDepthinX0(currentDepth_);
  if(HADSHOWER)
    origin=segiterator->positionAtDepthinL0(currentDepth_);

  //  std::cout << " Origin " << origin << std::endl;
  XYZVector zaxis(0,0,1);
  XYZVector planeVec1=(zaxis.Cross(particleDirection)).Unit();
  
  locToGlobal_=Transform3D(Point(0,0,0),
			   Point(0,0,1),
			   Point(1,0,0),
			   (Point)origin,
			   (Point)(origin+particleDirection),
			   (Point)(origin+planeVec1));

  return true;
}
