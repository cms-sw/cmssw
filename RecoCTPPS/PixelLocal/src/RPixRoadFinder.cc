
#include "RecoCTPPS/PixelLocal/interface/RPixRoadFinder.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TMatrixD.h"

#include <vector>
#include <memory>
#include <string>
#include <iostream>

//------------------------------------------------------------------------------------------------//

RPixRoadFinder::RPixRoadFinder(edm::ParameterSet const& parameterSet) : 
	RPixDetPatternFinder(parameterSet){

	verbosity_ = parameterSet.getUntrackedParameter<int> ("verbosity");
  roadRadius_ = parameterSet.getParameter<double>("roadRadius");
  minRoadSize_ = parameterSet.getParameter<int>("minRoadSize");
  maxRoadSize_ = parameterSet.getParameter<int>("maxRoadSize");

}

//------------------------------------------------------------------------------------------------//

RPixRoadFinder::~RPixRoadFinder(){
}

//------------------------------------------------------------------------------------------------//

void RPixRoadFinder::findPattern(){

  Road temp_all_hits;
  temp_all_hits.clear();

// convert local hit sto global and push them to a vector
  for(const auto & ds_rh2 : hitVector_){
    uint32_t myid = ds_rh2.id;
    for (const auto & _rh : ds_rh2.data){
      PointInPlane thePointAndRecHit;
      thePointAndRecHit.recHit=_rh; 
      CLHEP::Hep3Vector localV(_rh.getPoint().x(),_rh.getPoint().y(),_rh.getPoint().z() );
      CLHEP::Hep3Vector globalV = geometry_.localToGlobal(ds_rh2.id,localV);
      thePointAndRecHit.globalPoint=globalV;
      TMatrixD localError(3,3);
      localError[0][0] = _rh.getError().xx();
      localError[0][1] = _rh.getError().xy();
      localError[0][2] =                  0.;
      localError[1][0] = _rh.getError().xy();
      localError[1][1] = _rh.getError().yy();
      localError[1][2] =                  0.;
      localError[2][0] =                  0.;
      localError[2][1] =                  0.;
      localError[2][2] =                  0.;
      if(verbosity_>2) std::cout<<"Hits = "<<ds_rh2.data.size()<<std::endl;
      TMatrixD theRotationTMatrix(planeRotationMatrixMap_[CTPPSPixelDetId(ds_rh2.id)]);

      TMatrixD theRotationTMatrixInverted(theRotationTMatrix);
      theRotationTMatrixInverted.Invert();
      TMatrixD globalError = theRotationTMatrixInverted * localError * theRotationTMatrix;
      thePointAndRecHit.globalError.ResizeTo(3,3);
      thePointAndRecHit.globalError=globalError;
      thePointAndRecHit.detId = myid;
      temp_all_hits.push_back(thePointAndRecHit);
    }

  }
  
  Road::iterator _gh1 = temp_all_hits.begin();
  Road::iterator _gh2;

  patternVector_.clear();

//look for points near wrt each other
// starting algorithm
  while( _gh1 != temp_all_hits.end() && temp_all_hits.size() > minRoadSize_){
    Road temp_road;
  
    _gh2 = _gh1;

    CLHEP::Hep3Vector currPoint = _gh1->globalPoint;
    CTPPSPixelDetId currDet = CTPPSPixelDetId(_gh1->detId);
    if(verbosity_>1)  std::cout << " current point " << currPoint << std::endl;
    while( _gh2 != temp_all_hits.end()){
      bool same_pot = false;
      CTPPSPixelDetId tmpGh2Id = CTPPSPixelDetId(_gh2->detId);
      if(    currDet.arm() == tmpGh2Id.arm() && currDet.station() == tmpGh2Id.station() && currDet.rp() == tmpGh2Id.rp() )same_pot = true;
      CLHEP::Hep3Vector subtraction = currPoint - _gh2->globalPoint;
      if(verbosity_>1) std::cout << "             Subtraction " << currPoint << " - " << _gh2->globalPoint << " " << subtraction.perp() << std::endl;
      if(subtraction.perp() < roadRadius_ && same_pot) {  /// 1mm
        temp_road.push_back(*_gh2);
        temp_all_hits.erase(_gh2);
      }else{
        ++_gh2;
      }
      if(verbosity_>1)std::cout << " SIZE " << temp_all_hits.size() <<std::endl;
    }

    if(temp_road.size() > minRoadSize_ && temp_road.size() < maxRoadSize_ )patternVector_.push_back(temp_road);
   
  }
// end of algorithm

  if(verbosity_)std::cout << "+-+-+-+-+-+-    Number of pseudo tracks " << patternVector_.size() <<std::endl;

  if(!patternVector_.empty())
    for (auto const & ptrack : patternVector_){

      if(verbosity_) std::cout << "     ptrack size = "<<ptrack.size() << std::endl;
    }

}


