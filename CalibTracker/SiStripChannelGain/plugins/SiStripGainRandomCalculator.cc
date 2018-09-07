// -*- C++ -*-
// Package:    SiStripChannelGain
// Class:      SiStripGainRandomCalculator
// Original Author:  G. Bruno
//         Created:  Mon May 20 10:04:31 CET 2007

#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainRandomCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "CLHEP/Random/RandGauss.h"


using namespace cms;
using namespace std;


SiStripGainRandomCalculator::SiStripGainRandomCalculator(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripApvGain>(iConfig), m_cacheID_(0){

  
  edm::LogInfo("SiStripGainRandomCalculator::SiStripGainRandomCalculator");

//   std::string Mode=iConfig.getParameter<std::string>("Mode");
//   if (Mode==std::string("Gaussian")) GaussianMode_=true;
//   else if (IOVMode==std::string("Constant")) ConstantMode_=true;
//   else  edm::LogError("SiStripGainRandomCalculator::SiStripGainRandomCalculator(): ERROR - unknown generation mode...will not store anything on the DB") << std::endl;

  detid_apvs_.clear();

  meanGain_=iConfig.getParameter<double>("MeanGain");
  sigmaGain_=iConfig.getParameter<double>("SigmaGain");
  minimumPosValue_=iConfig.getParameter<double>("MinPositiveGain");
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);


}


SiStripGainRandomCalculator::~SiStripGainRandomCalculator(){

   edm::LogInfo("SiStripGainRandomCalculator::~SiStripGainRandomCalculator");
}



void SiStripGainRandomCalculator::algoAnalyze(const edm::Event & event, const edm::EventSetup& iSetup){


  unsigned long long cacheID = iSetup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  
  if (m_cacheID_ != cacheID) {
    
    m_cacheID_ = cacheID; 

    edm::ESHandle<TrackerGeometry> pDD;

    iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
    edm::LogInfo("SiStripGainRandomCalculator::algoAnalyze - got new geometry  ")<<std::endl;

    detid_apvs_.clear();
    
    edm::LogInfo("SiStripGainCalculator") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
    
    for( const auto& it : pDD->detUnits()) {
  
      if( dynamic_cast<const StripGeomDetUnit*>(it)!=nullptr){
	uint32_t detid=(it->geographicalId()).rawId();            
	const StripTopology & p = dynamic_cast<const StripGeomDetUnit*>(it)->specificTopology();
	unsigned short NAPVs = p.nstrips()/128;
	if(NAPVs<1 || NAPVs>6 ) {
	  edm::LogError("SiStripGainCalculator")<<" Problem with Number of strips in detector.. "<< p.nstrips() <<" Exiting program"<<endl;
	  exit(1);
	}
	detid_apvs_.push_back( pair<uint32_t,unsigned short>(detid,NAPVs) );
	if (printdebug_)
	  edm::LogInfo("SiStripGainCalculator")<< "detid " << detid << " apvs " << NAPVs;
      }
    }
  }


}


std::unique_ptr<SiStripApvGain> SiStripGainRandomCalculator::getNewObject() {

  std::cout<<"SiStripGainRandomCalculator::getNewObject called"<<std::endl;

  auto obj = std::make_unique<SiStripApvGain>();

  for(std::vector< pair<uint32_t,unsigned short> >::const_iterator it = detid_apvs_.begin(); it != detid_apvs_.end(); it++){
    //Generate Gain for det detid
    std::vector<float> theSiStripVector;
    for(unsigned short j=0; j<it->second; j++){
      float gain;

      //      if(sigmaGain_/meanGain_ < 0.00001) gain = meanGain_;
      //      else{
      gain = CLHEP::RandGauss::shoot(meanGain_, sigmaGain_);
      if(gain<=minimumPosValue_) gain=minimumPosValue_;
      //      }

      if (printdebug_)
	edm::LogInfo("SiStripGainCalculator") << "detid " << it->first << " \t"
					      << " apv " << j << " \t"
					      << gain    << " \t" 
					      << std::endl; 	    
      theSiStripVector.push_back(gain);
    }
    
    
    SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(it->first,range) )
      edm::LogError("SiStripGainCalculator")<<"[SiStripGainCalculator::beginJob] detid already exists"<<std::endl;

  }
  
  return obj;

}


