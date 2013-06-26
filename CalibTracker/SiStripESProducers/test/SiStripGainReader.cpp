
// system include files
//#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//#include "CalibTracker/SiStripGainESProducer/test/SiStripGainReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>


//
//
// class decleration
//
//namespace cms{

  class SiStripGainReader : public edm::EDAnalyzer {

  public:
    explicit SiStripGainReader( const edm::ParameterSet& );
    ~SiStripGainReader();
  
    void analyze( const edm::Event&, const edm::EventSetup& );

  private:
    bool printdebug_;
  };
//}

//using namespace cms;

SiStripGainReader::SiStripGainReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)){}

SiStripGainReader::~SiStripGainReader(){}

void SiStripGainReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  edm::ESHandle<SiStripGain> SiStripGain_;
  iSetup.get<SiStripGainRcd>().get(SiStripGain_);
  edm::LogInfo("SiStripGainReader") << "[SiStripGainReader::analyze] End Reading SiStripGain" << std::endl;
  
  std::vector<uint32_t> detid;
  SiStripGain_->getDetIds(detid);

  SiStripApvGain::Range range=SiStripGain_->getRange(detid[0]);

  edm::LogInfo("Number of detids ")  << detid.size() << std::endl;
  int apv=0;
  edm::LogInfo(" First det gain values  ")  <<  std::endl; 	
  for(int it=0;it<range.second-range.first;it++){
    edm::LogInfo("SiStripApvGainReader")  << "detid " << detid[0] << " \t"
					     << " apv " << apv++ << " \t"
					     << SiStripGain_->getApvGain(it,range)     << " \t" 
					     << std::endl; 
  }


  
  if (printdebug_)
    for (size_t id=0;id<detid.size();id++)
      {
	SiStripApvGain::Range range=SiStripGain_->getRange(detid[id]);
	
	apv=0;
	for(int it=0;it<range.second-range.first;it++){
	  edm::LogInfo("SiStripGainReader")  << "detid " << detid[id] << " \t"
					     << " apv " << apv++ << " \t"
					     << SiStripGain_->getApvGain(it,range)     << " \t" 
					     << std::endl; 	    
	} 
      }
}

DEFINE_FWK_MODULE(SiStripGainReader);


