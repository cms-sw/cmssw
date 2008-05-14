// -*- C++ -*-
// Package:    SiStripPedestals
// Class:      SiStripNoiseDummyCalculator
// Original Author:  G. Bruno
//         Created:  Mon May 20 10:04:31 CET 2007
// $Id: SiStripNoiseDummyCalculator.cc,v 1.3 2008/03/04 15:59:23 giordano Exp $

#include "CalibTracker/SiStripESProducers/plugins/geom/SiStripNoiseDummyCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"


using namespace cms;
using namespace std;

SiStripNoiseDummyCalculator::SiStripNoiseDummyCalculator(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripNoises>::ConditionDBWriter<SiStripNoises>(iConfig), m_cacheID_(0){

  
  edm::LogInfo("SiStripNoiseDummyCalculator::SiStripNoiseDummyCalculator");

  detData_.clear();

  stripLengthMode_ = iConfig.getParameter<bool>("StripLengthMode");
  
  noiseStripLengthLinearSlope_ = iConfig.getParameter<double>("NoiseStripLengthSlope");  
  noiseStripLengthLinearQuote_ = iConfig.getParameter<double>("NoiseStripLengthQuote");  
  electronsPerADC_ = iConfig.getParameter<double>("electronPerAdc");


  meanNoise_=iConfig.getParameter<double>("MeanNoise");
  sigmaNoise_=iConfig.getParameter<double>("SigmaNoise");
  minimumPosValue_=iConfig.getParameter<double>("MinPositiveNoise");

  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);

}


SiStripNoiseDummyCalculator::~SiStripNoiseDummyCalculator(){

   edm::LogInfo("SiStripNoiseDummyCalculator::~SiStripNoiseDummyCalculator");
}



void SiStripNoiseDummyCalculator::algoAnalyze(const edm::Event & event, const edm::EventSetup& iSetup){

  unsigned long long cacheID = iSetup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  
  if (m_cacheID_ != cacheID) {

    //remove
    //    mys.setESObjects(iSetup);

    
    m_cacheID_ = cacheID; 

    edm::ESHandle<TrackerGeometry> pDD;

    iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
    edm::LogInfo("SiStripNoiseDummyCalculator::algoAnalyze - got new geometry  ")<<std::endl;


    detData_.clear();
    
    edm::LogInfo("SiStripNoiseDummyCalculator") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
    
    for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
  
      const StripGeomDetUnit* mit = dynamic_cast<StripGeomDetUnit*>(*it);

      if(mit!=0){

	uint32_t detid=(mit->geographicalId()).rawId();
	double stripLength = mit->specificTopology().stripLength();
	unsigned short numberOfAPVPairs= mit->specificTopology().nstrips()/256;

	detData_[detid]=pair<unsigned short, double>(numberOfAPVPairs, stripLength);
	if (printdebug_)
	  edm::LogInfo("SiStripNoiseDummyCalculator")<< "detid: " << detid << " strip length: " << stripLength << "  number of APV pairs: " << numberOfAPVPairs;
      }
    }

    if (printdebug_) edm::LogInfo("SiStripNoiseDummyCalculator - Number of detectors for which got noise value is ")<< detData_.size();


  }

}


SiStripNoises * SiStripNoiseDummyCalculator::getNewObject() {

  std::cout<<"SiStripNoiseDummyCalculator::getNewObject called"<<std::endl;

  SiStripNoises * obj = new SiStripNoises();

  bool firstdet=true;

  for(std::map< uint32_t, pair<unsigned short, double> >::const_iterator it = detData_.begin(); it != detData_.end(); it++){
    //Generate Noise for det detid

    SiStripNoises::InputVector theSiStripVector;
    for(unsigned short j=0; j<(it->second).first; j++){
      for(int strip=0; strip<256; ++strip){

	float noise;

	if(stripLengthMode_){

	  noise = ( noiseStripLengthLinearSlope_ * (detData_[it->first].second) + noiseStripLengthLinearQuote_) / electronsPerADC_;


	}
	else{
	  noise = RandGauss::shoot(meanNoise_,sigmaNoise_);
	  if(noise<=minimumPosValue_) noise=minimumPosValue_;
	}

	if (printdebug_ && firstdet) {

	  edm::LogInfo("SiStripNoisesDummyCalculator") << "detid: " << it->first  << " strip: " << j*256+strip <<  " noise: " << noise     << " \t"   << std::endl; 	    


	}


	obj->setData(noise,theSiStripVector);

      }

    }	    
      
    firstdet=false;

    if ( ! obj->put(it->first,theSiStripVector) )
      edm::LogError("SiStripNoiseDummyCalculator")<<"[SiStripNoiseDummyCalculator::beginJob] detid already exists"<<std::endl;

  }



//   //remove
//  for(std::map< uint32_t, pair<unsigned short, double> >::const_iterator pit = detData_.begin(); pit != detData_.end(); pit++){

//    SiStripNoises::Range myra = obj->getRange((*pit).first);

//    for(unsigned short j=0; j<(pit->second).first; j++){
//      for(int strip=0; strip<256; ++strip){

//        uint16_t myst=j*256+strip;
       
//        float mysnoise = (static_cast<int16_t>  (mys.getNoise((*pit).first,  myst)*10.0 + 0.5) & 0x01FF)/10.0;


//        //       if (mys.getNoise((*pit).first,  myst) + 0.00001 < obj->getNoise( myst , myra) || mys.getNoise((*pit).first,  myst) - 0.00001 > obj->getNoise( myst , myra)  ) edm::LogError("ERROR")<< "DETID: "<< (*pit).first << " STRIP: "<< myst <<" NOISE SERVICE NOISE: "<< mys.getNoise((*pit).first,  myst) <<  " MY NOISE: "<<  obj->getNoise( myst , myra) << endl;

//        if (mysnoise + 0.00001 < obj->getNoise( myst , myra) || mysnoise - 0.00001 > obj->getNoise( myst , myra)  ) edm::LogError("ERROR")<< "DETID: "<< (*pit).first << " STRIP: "<< myst <<" NOISE SERVICE NOISE: "<< mys.getNoise((*pit).first,  myst) <<  " MY NOISE: "<<  obj->getNoise( myst , myra) << " NOISE SERVICE MODIFIED"<< mysnoise<< endl;

       

//      }
//    }

//  }
//  //end remove


  
  return obj;

}


