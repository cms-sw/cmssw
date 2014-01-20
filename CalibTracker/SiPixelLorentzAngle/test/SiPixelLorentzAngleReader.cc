#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/SiPixelLorentzAngle/test/SiPixelLorentzAngleReader.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h" 

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

 if(useSimRcd_ == true) {
   iSetup.get<SiPixelLorentzAngleSimRcd>().get(SiPixelLorentzAngle_);
   std::cout<<" Show LA for simulations "<<std::endl;
 } else {
   iSetup.get<SiPixelLorentzAngleRcd>().get(SiPixelLorentzAngle_);
   std::cout<<" Show LA for reconstruction "<<std::endl;
 }
  
edm::LogInfo("SiPixelLorentzAngleReader") << "[SiPixelLorentzAngleReader::analyze] End Reading SiPixelLorentzAngle" << std::endl;
  edm::Service<TFileService> fs;
  LorentzAngleBarrel_ = fs->make<TH1F>("LorentzAngleBarrelPixel","LorentzAngleBarrelPixel",150,0,0.15);
  LorentzAngleForward_= fs->make<TH1F>("LorentzAngleForwardPixel","LorentzAngleForwardPixel",150,0,0.15);
  std::map<unsigned int,float> detid_la= SiPixelLorentzAngle_->getLorentzAngles();
  std::map<unsigned int,float>::const_iterator it;
  double la_old=-1.;
  for (it=detid_la.begin();it!=detid_la.end();it++) {
    //if(printdebug_) std::cout  << "detid " << it->first << " \t" << " Lorentz angle  " << it->second  << std::endl;
    //if(printdebug_) edm::LogInfo("SiPixelLorentzAngleReader")  << "detid " << it->first << " \t" << " Lorentz angle  " << it->second;

    unsigned int subdet   = DetId(it->first).subdetId();
    int detid = it->first;

    if(subdet == static_cast<int>(PixelSubdetector::PixelBarrel)){
      LorentzAngleBarrel_->Fill(it->second);
      //std::cout  << " bpix detid " << it->first << " \t" << " Lorentz angle  " << it->second  << std::endl;
      //edm::LogInfo("SiPixelLorentzAngleReader")  << " bpix detid " << it->first << " \t" << " Lorentz angle  " << it->second;

      PXBDetId pdetId = PXBDetId(detid);
      //unsigned int detTypeP=pdetId.det();
      //unsigned int subidP=pdetId.subdetId();
      // Barell layer = 1,2,3
      int layerC=pdetId.layer();
      // Barrel ladder id 1-20,32,44.
      int ladderC=pdetId.ladder();
      // Barrel Z-index=1,8
      int zindex=pdetId.module();

      if(printdebug_) {

	std::cout<<"BPix - layer "<<layerC<<" ladder "<<ladderC<<" ring "<<zindex<< " Lorentz angle  " << it->second  << std::endl;
	edm::LogInfo("SiPixelLorentzAngleReader")  <<"BPix - layer "<<layerC<<" ladder "<<ladderC<<" ring "<<zindex<< " Lorentz angle  " << it->second;

      } else {

	if(ladderC==1) { // print once per ring 

	  if(it->second != la_old) {
	    std::cout<<"BPix - layer "<<layerC<<" ladder "<<ladderC<<" ring "<<zindex<< " Lorentz angle  " << it->second  << std::endl;
	    edm::LogInfo("SiPixelLorentzAngleReader")  <<"BPix - layer "<<layerC<<" ladder "<<ladderC<<" ring "<<zindex<< " Lorentz angle  " << it->second;
	  } // else {std::cout<<"same"<<std::endl;}

	  la_old = it->second;
	}
      }

    }else if(subdet == static_cast<int>(PixelSubdetector::PixelEndcap)){
      LorentzAngleForward_->Fill(it->second);

      PXFDetId pdetId = PXFDetId(detid);       
      int disk=pdetId.disk(); //1,2,3
      int blade=pdetId.blade(); //1-24
      int moduleF=pdetId.module(); //
      int side=pdetId.side(); //size=1 for -z, 2 for +z
      int panel=pdetId.panel(); //panel=1

      if(blade==1 && moduleF==1 && side==1 && panel==1) { // print once per disk 
	std::cout<<"FPix - disk "<<disk<< " Lorentz angle  " << it->second  << std::endl;
	edm::LogInfo("SiPixelLorentzAngleReader")  <<"FPix - disk "<<disk<< " Lorentz angle  " << it->second;
      }

    }
  } 
}
