#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CondTools/SiPixel/test/SiPixelLorentzAngleReader.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>


using namespace cms;

SiPixelLorentzAngleReader::SiPixelLorentzAngleReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)),
  useSimRcd_( iConfig.getParameter<bool>("useSimRcd") )
{
}

SiPixelLorentzAngleReader::~SiPixelLorentzAngleReader(){}

void SiPixelLorentzAngleReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){
 edm::ESHandle<SiPixelLorentzAngle> SiPixelLorentzAngle_; 
 if(useSimRcd_ == true)
  iSetup.get<SiPixelLorentzAngleSimRcd>().get(SiPixelLorentzAngle_);
 else
  iSetup.get<SiPixelLorentzAngleRcd>().get(SiPixelLorentzAngle_);
  edm::LogInfo("SiPixelLorentzAngleReader") << "[SiPixelLorentzAngleReader::analyze] End Reading SiPixelLorentzAngle" << std::endl;
  edm::Service<TFileService> fs;
  LorentzAngleBarrel_ = fs->make<TH1F>("LorentzAngleBarrelPixel","LorentzAngleBarrelPixel",150,0,0.15);
  LorentzAngleForward_= fs->make<TH1F>("LorentzAngleForwardPixel","LorentzAngleForwardPixel",150,0,0.15);
  std::map<unsigned int,float> detid_la= SiPixelLorentzAngle_->getLorentzAngles();
  std::map<unsigned int,float>::const_iterator it;
  for (it=detid_la.begin();it!=detid_la.end();it++)
      {
	  		//	std::cout  << "detid " << it->first << " \t" << " Lorentz angle  " << it->second  << std::endl;
			//edm::LogInfo("SiPixelLorentzAngleReader")  << "detid " << it->first << " \t" << " Lorentz angle  " << it->second;
			unsigned int subdet   = DetId(it->first).subdetId();
			if(subdet == static_cast<int>(PixelSubdetector::PixelBarrel)){
				LorentzAngleBarrel_->Fill(it->second);
			}else if(subdet == static_cast<int>(PixelSubdetector::PixelEndcap)){
				LorentzAngleForward_->Fill(it->second);
			}
      } 
}
