#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "CondTools/SiStrip/plugins/SiStripDetVOffReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>

SiStripDetVOffReader::SiStripDetVOffReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",true)){}


SiStripDetVOffReader::~SiStripDetVOffReader(){}


void SiStripDetVOffReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  edm::ESHandle<SiStripDetVOff> SiStripDetVOff_;
  iSetup.get<SiStripDetVOffRcd>().get(SiStripDetVOff_);
  edm::LogInfo("SiStripDetVOffReader") << "[SiStripDetVOffReader::analyze] End Reading SiStripDetVOff" << std::endl;
  

  // put here a vector of DetIds to compare 
  // Here we just take the vector with all modules that have HV OFF
  
  // replace this code, with Your own detids
  std::vector <uint32_t> detid;
  SiStripDetVOff_->getDetIds(detid);
  //


  if (printdebug_){
    for (uint32_t  id=0;id<=detid.size();id++)
      {
	bool hvflag=SiStripDetVOff_->IsModuleHVOff(detid[id]);
	bool lvflag=SiStripDetVOff_->IsModuleLVOff(detid[id]);
        bool vflag =SiStripDetVOff_->IsModuleVOff(detid[id]);
	if(hvflag==true) {
          edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " HV\t OFF\n";
        }
	else {
          edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " HV\t ON\n";
        }
	if(lvflag==true) {
          edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " LV\t OFF\n";
        }
	else {
          edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " LV\t ON\n";
        }
	if(vflag==true) {
          edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " V\t OFF\n";
        }
	else {
          edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " V\t ON\n";
        }
      } 
  }

}
