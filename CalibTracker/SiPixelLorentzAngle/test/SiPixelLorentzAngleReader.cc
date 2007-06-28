#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"

#include "CalibTracker/SiPixelLorentzAngle/test/SiPixelLorentzAngleReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>


using namespace cms;

SiPixelLorentzAngleReader::SiPixelLorentzAngleReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)){}

SiPixelLorentzAngleReader::~SiPixelLorentzAngleReader(){}

void SiPixelLorentzAngleReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){
  
  edm::ESHandle<SiPixelLorentzAngle> SiPixelLorentzAngle_;
  iSetup.get<SiPixelLorentzAngleRcd>().get(SiPixelLorentzAngle_);
  edm::LogInfo("SiPixelLorentzAngleReader") << "[SiPixelLorentzAngleReader::analyze] End Reading SiPixelLorentzAngle" << std::endl;
  
  std::map<unsigned int,float> detid_la= SiPixelLorentzAngle_->getLorentzAngles();
  std::map<unsigned int,float>::const_iterator it;
  for (it=detid_la.begin();it!=detid_la.end();it++)
      {
	edm::LogInfo("SiPixelLorentzAngleReader")  << "detid " << it->first << " \t"
						   << " Lorentz angle  " << it->second;
      } 
}

