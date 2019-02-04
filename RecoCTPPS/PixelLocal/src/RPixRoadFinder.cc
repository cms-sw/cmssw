
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

#include "TMath.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

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
  for(const auto & ds_rh2 : *hitVector_){
    const auto myid = CTPPSPixelDetId(ds_rh2.id);
    for (const auto & it_rh : ds_rh2.data){
      CLHEP::Hep3Vector localV(it_rh.getPoint().x(),it_rh.getPoint().y(),it_rh.getPoint().z() );
      CLHEP::Hep3Vector globalV = geometry_->localToGlobal(ds_rh2.id,localV);
      math::Error<3>::type localError;
      localError[0][0] = it_rh.getError().xx();
      localError[0][1] = it_rh.getError().xy();
      localError[0][2] =                  0.;
      localError[1][0] = it_rh.getError().xy();
      localError[1][1] = it_rh.getError().yy();
      localError[1][2] =                  0.;
      localError[2][0] =                  0.;
      localError[2][1] =                  0.;
      localError[2][2] =                  0.;
      if(verbosity_>2) edm::LogInfo("RPixRoadFinder")<<"Hits = "<<ds_rh2.data.size();

      DetGeomDesc::RotationMatrix theRotationMatrix = geometry_->getSensor(myid)->rotation();
      AlgebraicMatrix33 theRotationTMatrix;
      theRotationMatrix.GetComponents(theRotationTMatrix(0, 0), theRotationTMatrix(0, 1), theRotationTMatrix(0, 2),
                                      theRotationTMatrix(1, 0), theRotationTMatrix(1, 1), theRotationTMatrix(1, 2),
                                      theRotationTMatrix(2, 0), theRotationTMatrix(2, 1), theRotationTMatrix(2, 2));

      math::Error<3>::type globalError = ROOT::Math::SimilarityT(theRotationTMatrix, localError);
      PointInPlane thePointAndRecHit = {globalV,globalError,it_rh,myid};
      temp_all_hits.push_back(thePointAndRecHit);
    }

  }
  
  Road::iterator it_gh1 = temp_all_hits.begin();
  Road::iterator it_gh2;

  patternVector_.clear();

//look for points near wrt each other
// starting algorithm
  while( it_gh1 != temp_all_hits.end() && temp_all_hits.size() >= minRoadSize_){
    Road temp_road;
  
    it_gh2 = it_gh1;

    CLHEP::Hep3Vector currPoint = it_gh1->globalPoint;
    CTPPSPixelDetId currDet = CTPPSPixelDetId(it_gh1->detId);

    while( it_gh2 != temp_all_hits.end()){
      bool same_pot = false;
      CTPPSPixelDetId tmpGh2Id = CTPPSPixelDetId(it_gh2->detId);
      if ( currDet.getRPId() == tmpGh2Id.getRPId() ) same_pot = true;
      CLHEP::Hep3Vector subtraction = currPoint - it_gh2->globalPoint;

      if(subtraction.perp() < roadRadius_ && same_pot) {  /// 1mm
        temp_road.push_back(*it_gh2);
        temp_all_hits.erase(it_gh2);
      }else{
        ++it_gh2;
      }

    }

    if(temp_road.size() >= minRoadSize_ && temp_road.size() < maxRoadSize_ )patternVector_.push_back(temp_road);
   
  }
// end of algorithm



}


