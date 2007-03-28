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
// $Id: SiPixelDigiSource.cc,v 1.10 2007/03/09 08:35:49 chiochia Exp $
//
//
#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiSource.h"
// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
// DQM Framework
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
// DataFormats
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
//
#include <boost/cstdint.hpp>
#include <string>
#include <stdlib.h>
using namespace std;

SiPixelDigiSource::SiPixelDigiSource(const edm::ParameterSet& iConfig) :
  conf_(iConfig)
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

  // retrieve producer name of input SiPixelDigiCollection
  std::string digiProducer = conf_.getUntrackedParameter<std::string>("DigiProducer","siPixelDigis");

  // get input data
  edm::Handle< edm::DetSetVector<PixelDigi> >  input;
  iEvent.getByLabel(digiProducer, input);

  std::map<uint32_t,SiPixelDigiModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    
    (*struct_iter).second->fill(*input);
    
  }


  // slow down...
  usleep(100000);
 
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelDigiSource::buildStructure(const edm::EventSetup& iSetup){

  std::cout <<" *** SiPixelDigiSource::buildStructure" << std::endl;
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
                                                                                                                                                       
  std::cout <<" *** Geometry node for TrackerGeom is  "<<&(*pDD)<<std::endl;
  std::cout <<" *** I have " << pDD->dets().size() <<" detectors"<<std::endl;
  std::cout <<" *** I have " << pDD->detTypes().size() <<" types"<<std::endl;
  
  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
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
  theDMBE->setVerbose(0);
    
  SiPixelFolderOrganizer theSiPixelFolder;
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    
    /// Create folder tree and book histograms 
    if(theSiPixelFolder.setModuleFolder((*struct_iter).first)){
      (*struct_iter).second->book();
    } else {
      throw cms::Exception("LogicError")
	<< "[SiPixelDigiSource::bookMEs] Creation of DQM folder failed";
    }
    
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelDigiSource);
