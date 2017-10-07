#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"

#include "CondTools/SiStrip/plugins/SiStripLorentzAngleReader.h"

#include <iostream>
#include <cstdio>
#include <sys/time.h>


using namespace cms;

SiStripLorentzAngleReader::SiStripLorentzAngleReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",5)),
  label_(iConfig.getUntrackedParameter<std::string>("label","")){}
SiStripLorentzAngleReader::~SiStripLorentzAngleReader(){}

void SiStripLorentzAngleReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){
  
  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle_;
  iSetup.get<SiStripLorentzAngleRcd>().get(label_,SiStripLorentzAngle_);
  edm::LogInfo("SiStripLorentzAngleReader") << "[SiStripLorentzAngleReader::analyze] End Reading SiStripLorentzAngle with label " << label_<< std::endl;
  
  std::map<unsigned int,float> detid_la= SiStripLorentzAngle_->getLorentzAngles();
  std::map<unsigned int,float>::const_iterator it;
  size_t count=0;
  for (it=detid_la.begin();it!=detid_la.end() && count<printdebug_;it++)
      {
	edm::LogInfo("SiStripLorentzAngleReader")  << "detid " << it->first << " \t"
						   << " Lorentz angle  " << it->second;
	count++;
      } 
}

