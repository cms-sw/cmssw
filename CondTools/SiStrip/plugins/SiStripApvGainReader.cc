#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "CondTools/SiStrip/plugins/SiStripApvGainReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>


using namespace cms;

SiStripApvGainReader::SiStripApvGainReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",true)),
  formatedOutput_(iConfig.getUntrackedParameter<std::string>("outputFile","")),
  gainType_ (iConfig.getUntrackedParameter<uint32_t>("gainType",1))
  {}

SiStripApvGainReader::~SiStripApvGainReader(){}

void SiStripApvGainReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  edm::ESHandle<SiStripGain> SiStripApvGain_;
  iSetup.get<SiStripGainRcd>().get(SiStripApvGain_);
  edm::LogInfo("SiStripApvGainReader") << "[SiStripApvGainReader::analyze] End Reading SiStripApvGain" << std::endl;
  
  std::vector<uint32_t> detid;
  SiStripApvGain_->getDetIds(detid);
  edm::LogInfo("Number of detids ")  << detid.size() << std::endl;

  FILE* pFile=NULL;
  if(formatedOutput_!="")pFile=fopen(formatedOutput_.c_str(), "w");

  for (size_t id=0;id<detid.size();id++){
    SiStripApvGain::Range range=SiStripApvGain_->getRange(detid[id], gainType_);	
    if(printdebug_){
       int apv=0;
       for(int it=0;it<range.second-range.first;it++){
          edm::LogInfo("SiStripApvGainReader")  << "detid " << detid[id] << " \t " << apv++ << " \t " << SiStripApvGain_->getApvGain(it,range)     << std::endl;        
       }
    }

    if(pFile){
       fprintf(pFile,"%i ",detid[id]);
       for(int it=0;it<range.second-range.first;it++){
          fprintf(pFile,"%f ", SiStripApvGain_->getApvGain(it,range) );
       }fprintf(pFile, "\n");
    }

  }

  if(pFile)fclose(pFile);
}
