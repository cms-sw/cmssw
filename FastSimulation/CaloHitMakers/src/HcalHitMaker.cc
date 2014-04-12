#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h" // PV
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

  // Watch out !!!! (Point) is a real point in the MathCore terminology (not a redefined a XYZPoint which
  // is actually a XYZVector in the MatchCore terminology). Therefore, the Transform3D is correctly applied
  point = locToGlobal_((Point)point);
  return addHit(point,layer);
}

bool HcalHitMaker::addHit(const XYZPoint& point, unsigned layer)
{
  // Temporary nasty hacks to avoid misbehaviour of not-intended-for-that
  //  getClosestCell in case of large (eta beyond HF ...)  and in EM showers 
  if(fabs(point.Z())>2000 || fabs(point.X())>2000 || fabs(point.Y())>2000) 
    { 
      if(EMSHOWER) 
	edm::LogWarning("HcalHitMaker") << "received a hit very far from the detector " << point << " coming from an electromagnetic shower. - Ignoring it" << std::endl;
      else if(HADSHOWER)
	edm::LogWarning("HcalHitMaker") << "received a hit very far from the detector " << point << " coming from a hadron shower. - Ignoring it" << std::endl;
      else if(MIP)
	edm::LogWarning("HcalHitMaker") << "received a hit very far from the detector " << point << " coming from a muon. - Ignoring it" << std::endl;
      return false; 
    } 


  double pointeta = fabs(point.eta());
  if(pointeta > 5.19) return false; 

  //calculate time of flight
  double dist = std::sqrt(point.X()*point.X() + point.Y()*point.Y() + point.Z()*point.Z());
  double tof = dist/29.98; //speed of light

  DetId thecellID(myCalorimeter->getClosestCell(point,false,false));
  
  HcalDetId myDetId(thecellID);
                                                                                                                                      
//   if ( myDetId.subdetId() == HcalForward ) {
//     std::cout << "HcalHitMaker : " << point.Z() << " " << myDetId.depth()    << std::endl;
//   }
                                                                                                                                      
//   std::cout << "BEFORE" << std::endl;
//   std::cout << "HcalHitMaker : subdetId : " << myDetId.subdetId() << std::endl;
//   std::cout << "HcalHitMaker : depth    : " << myDetId.depth()    << std::endl;
//   std::cout << "HcalHitMaker : ieta     : " << myDetId.ieta()     << std::endl;
//   std::cout << "HcalHitMaker : iphi     : " << myDetId.iphi()     << std::endl;
//   std::cout << "HcalHitMaker : spotE    : " << spotEnergy         << std::endl;
//   std::cout << "HcalHitMaker : point.X  : " << point.X()          << std::endl;
//   std::cout << "HcalHitMaker : point.Y  : " << point.Y()          << std::endl;
//   std::cout << "HcalHitMaker : point.Z  : " << point.Z()          << std::endl;
                                                                                                                                      
  if ( myDetId.subdetId() == HcalForward ) {
    int mylayer = layer;
    if ( myDetId.depth()==2 ) {
      mylayer = (int)layer;
    } else {
      mylayer = 1;
    }
    HcalDetId myDetId2((HcalSubdetector)myDetId.subdetId(),myDetId.ieta(),myDetId.iphi(),mylayer);
    thecellID = myDetId2;
	myDetId = myDetId2;
  }


  
  if(!thecellID.null()  && myDetId.depth()>0)
    {	
      CaloHitID current_id(thecellID.rawId(),tof,0); //no track yet
      
      //      std::cout << " FamosHcalHitMaker::addHit - the cell num " << cell
      //      		<< std::endl;

      std::map<CaloHitID,float>::iterator cellitr;
      cellitr = hitMap_.find(current_id);
      if(cellitr==hitMap_.end())
	{
	  hitMap_.insert(std::pair<CaloHitID,float>(current_id,spotEnergy));
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
HcalHitMaker::setDepth(double depth,bool inCm)
{
  currentDepth_=depth;
  std::vector<CaloSegment>::const_iterator segiterator;
  if(inCm)
    {
      segiterator = find_if(myGrid.getSegments().begin(),myGrid.getSegments().end(),CaloSegment::inSegment(currentDepth_));
    }
  else
    {
      if(EMSHOWER)
	segiterator = find_if(myGrid.getSegments().begin(),myGrid.getSegments().end(),CaloSegment::inX0Segment(currentDepth_));
      
      //Hadron shower 
      if(HADSHOWER)
	segiterator = find_if(myGrid.getSegments().begin(),myGrid.getSegments().end(),CaloSegment::inL0Segment(currentDepth_));
    }
  
  if(segiterator==myGrid.getSegments().end()) 
    {
      // Special trick  - As advised by Salavat, no leakage should be simulated
      if(depth > myGrid.getSegments().back().sL0Exit())
	{
	  segiterator= find_if(myGrid.getSegments().begin(),myGrid.getSegments().end(),CaloSegment::inL0Segment(myGrid.getSegments().back().sL0Exit()-1.));
	  depth=segiterator->sL0Exit()-1.;
	  currentDepth_=depth;
	  if(segiterator==myGrid.getSegments().end())
	    {
	      std::cout << " Could not go at such depth " << EMSHOWER << "  " << currentDepth_ << std::endl;
	      std::cout << " Track " << *(myGrid.getFSimTrack()) << std::endl;
	      return false;
	    }
	}
      else
	{
	  std::cout << " Could not go at such depth " << EMSHOWER << "  " << currentDepth_ << " " << myGrid.getSegments().back().sL0Exit() << std::endl; 
	  std::cout << " Track " << *(myGrid.getFSimTrack()) << std::endl; 
	  return false; 
	}
    }


  XYZPoint origin;
  if(inCm)
    {
      origin=segiterator->positionAtDepthincm(currentDepth_);
    }
  else
    {
      if(EMSHOWER)
	origin=segiterator->positionAtDepthinX0(currentDepth_);
      if(HADSHOWER)
	origin=segiterator->positionAtDepthinL0(currentDepth_);
    }
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
