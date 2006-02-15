// -*- C++ -*-
//
// Package:    SiPixelMonitorDigi
// Class:      SiPixelDigiSource
// 
/**\class 

 Description: Pixel DQM source for Digis

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id$
//
//
#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiSource.h"
// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
// DQM Framework
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
// Geometry
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetUnit.h"
// DataFormats
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
//
#include <boost/cstdint.hpp>
#include <string>
#include <stdlib.h>
using namespace std;

SiPixelDigiSource::SiPixelDigiSource(const edm::ParameterSet& iConfig)
{

   //now do what ever initialization is needed
   theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
   cout<<endl<<"SiPixelDigiSource::SiPixelDigiSource: BackEnd interface"<<endl;
}


SiPixelDigiSource::~SiPixelDigiSource()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


void SiPixelDigiSource::beginJob(const edm::EventSetup& iSetup){

  std::cout << " SiPixelDigiSource::beginJob - Initialisation ..... " << std::endl;
  eventNo = 0;
  // Build map
  buildStructure(iSetup);
  // Book Monitoring Elements
  bookMEs();

}


void SiPixelDigiSource::endJob(void){
  std::cout << " SiPixelDigiSource::endJob - Saving Root File " << std::endl;
  theDMBE->save("sourceoutputfile.root");
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void
SiPixelDigiSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;
  std::cout << " Processing event: " << eventNo << std::endl;
  edm::Handle<PixelDigiCollection> digiCollection;
  iEvent.getByLabel("pixdigi", digiCollection);
  
  std::map<uint32_t,SiPixelDigiModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    
    (*struct_iter).second->fill(digiCollection.product());
    // slow down...

    
  }
  
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelDigiSource::buildStructure(const edm::EventSetup& iSetup){

  //
  // Later on the structure will be built from cabling and
  // *not* from geometry
  //
  std::cout <<" *** SiPixelDigiSource::buildStructure" << std::endl;
  edm::ESHandle<TrackingGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
                                                                                                                                                       
  std::cout <<" *** Geometry node for TrackerGeom is  "<<&(*pDD)<<std::endl;
  std::cout <<" *** I have " << pDD->dets().size() <<" detectors"<<std::endl;
  std::cout <<" *** I have " << pDD->detTypes().size() <<" types"<<std::endl;
  
  for(TrackingGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
      DetId detId = (*it)->geographicalId();
      
      if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        //cout << " ---> Adding Barrel Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelDigiModule* pippo = new SiPixelDigiModule(id);
	thePixelStructure.insert(pair<uint32_t,SiPixelDigiModule*> (id,pippo));

      }	else if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
	//cout << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelDigiModule* pippo = new SiPixelDigiModule(id);
	thePixelStructure.insert(pair<uint32_t,SiPixelDigiModule*> (id,pippo));
      }

    }
  }
  cout << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelDigiSource::bookMEs(){

  std::map<uint32_t,SiPixelDigiModule*>::iterator struct_iter;
  string rootDir = "Tracker";
  theDMBE->setVerbose(0);

  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    if(DetId::DetId((*struct_iter).first).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
      int layer  = PXBDetId::PXBDetId((*struct_iter).first).layer();
      int ladder = PXBDetId::PXBDetId((*struct_iter).first).ladder();
      int module = PXBDetId::PXBDetId((*struct_iter).first).det();
      
      string ssubdet = "PixelBarrel"; 
      char slayer[80];  sprintf(slayer, "Layer_%i",layer);
      char sladder[80]; sprintf(sladder,"Ladder_%02i",ladder);
      char smodule[80]; sprintf(smodule,"Module_%i",module);
      string sfolder = rootDir + "/" + ssubdet + "/" + slayer + "/" + sladder + "/" + smodule;
      theDMBE->setCurrentFolder(sfolder.c_str());
      (*struct_iter).second->book();

    } else 
      if(DetId::DetId((*struct_iter).first).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
      string ssubdet = "PixelEndcap";
      int side   =  PXFDetId::PXFDetId((*struct_iter).first).side();
      int disk   =  PXFDetId::PXFDetId((*struct_iter).first).disk();
      int blade  =  PXFDetId::PXFDetId((*struct_iter).first).blade();
      //int panel  =  PXFDetId::PXFDetId((*struct_iter).first).panel();
      int module =  PXFDetId::PXFDetId((*struct_iter).first).det();
      char sside[80];  sprintf(sside, "Side_%i",side);
      char sdisk[80];  sprintf(sdisk, "Disk_%i",disk);
      char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
      //char spanel[80]; sprintf(spanel, "Panel_%i",panel);
      char smodule[80];sprintf(smodule,"Module_%i",module);
      string sfolder = rootDir + "/" + ssubdet + "/" + sside + "/" + sdisk + "/" + sblade + "/" + smodule;
      theDMBE->setCurrentFolder(sfolder.c_str());
      (*struct_iter).second->book();
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelDigiSource)
