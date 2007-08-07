// -*- C++ -*-
//
// Package:    SiPixelMonitorRawData
// Class:      SiPixelRawDataErrorSource
// 
/**\class 

 Description: 
 Produces histograms for error information generated at the raw2digi stage for the 
 pixel tracker.

 Implementation:
 Takes a DetSetVector<SiPixelRawDataError> as input, and uses it to populate  a folder 
 hierarchy (organized by detId) with histograms containing information about 
 the errors.  Uses SiPixelRawDataErrorModule class to book and fill individual folders.  
 Note that this source is different than other DQM sources in the creation of an 
 unphysical detId folder (detId=0xffffffff) to hold information about errors for which 
 there is no detId available (except the dummy detId given to it at raw2digi).

*/
//
// Original Author:  Andrew York
//
#include "DQM/SiPixelMonitorRawData/interface/SiPixelRawDataErrorSource.h"
// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// DQM Framework
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/Common/interface/DetSetVector.h"
//
#include <boost/cstdint.hpp>
#include <string>
#include <stdlib.h>

using namespace std;
using namespace edm;

SiPixelRawDataErrorSource::SiPixelRawDataErrorSource(const edm::ParameterSet& iConfig) :
  conf_(iConfig),
  src_( conf_.getParameter<edm::InputTag>( "src" ) )
{
   theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
   LogInfo ("PixelDQM") << "SiPixelRawDataErrorSource::SiPixelRawDataErrorSource: Got DQM BackEnd interface"<<endl;
}


SiPixelRawDataErrorSource::~SiPixelRawDataErrorSource()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelRawDataErrorSource::~SiPixelRawDataErrorSource: Destructor"<<endl;
}


void SiPixelRawDataErrorSource::beginJob(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") << " SiPixelRawDataErrorSource::beginJob - Initialisation ... " << std::endl;
  eventNo = 0;
  // Build map
  buildStructure(iSetup);
  // Book Monitoring Elements
  bookMEs();

}


void SiPixelRawDataErrorSource::endJob(void){
  LogInfo ("PixelDQM") << " SiPixelRawDataErrorSource::endJob - Saving Root File " << std::endl;
  std::string outputFile = conf_.getParameter<std::string>("outputFile");
  theDMBE->save( outputFile.c_str() );
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void
SiPixelRawDataErrorSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;

  // get input data
  edm::Handle< edm::DetSetVector<SiPixelRawDataError> >  input;
  iEvent.getByLabel( src_, input );

  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    
    (*struct_iter).second->fill(*input);
    
  }

  

  // slow down...
  usleep(100000);
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelRawDataErrorSource::buildStructure(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") <<" SiPixelRawDataErrorSource::buildStructure" ;
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

  LogVerbatim ("PixelDQM") << " *** Geometry node for TrackerGeom is  "<<&(*pDD)<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->dets().size() <<" detectors"<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->detTypes().size() <<" types"<<std::endl;
  
  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    
    if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){

      DetId detId = (*it)->geographicalId();
      const GeomDetUnit      * geoUnit = pDD->idToDetUnit( detId );
      const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      int nrows = (pixDet->specificTopology()).nrows();
      int ncols = (pixDet->specificTopology()).ncolumns();

      if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        LogDebug ("PixelDQM") << " ---> Adding Barrel Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelRawDataErrorModule* theModule = new SiPixelRawDataErrorModule(id, ncols, nrows);
	thePixelStructure.insert(pair<uint32_t,SiPixelRawDataErrorModule*> (id,theModule));

      }	else if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
	LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelRawDataErrorModule* theModule = new SiPixelRawDataErrorModule(id, ncols, nrows);
	thePixelStructure.insert(pair<uint32_t,SiPixelRawDataErrorModule*> (id,theModule));
      }
    }
  }

  LogDebug ("PixelDQM") << " ---> Adding Module for Additional Errors" << endl;
  uint32_t id = 0xffffffff;
  SiPixelRawDataErrorModule* theModule = new SiPixelRawDataErrorModule(id);
  thePixelStructure.insert(pair<uint32_t,SiPixelRawDataErrorModule*> (id,theModule));
  
  LogInfo ("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelRawDataErrorSource::bookMEs(){
  
  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter;
  theDMBE->setVerbose(0);
  
  SiPixelFolderOrganizer theSiPixelFolder;
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    
    /// Create folder tree and book histograms 
    if(theSiPixelFolder.setModuleFolder((*struct_iter).first)){
      if((*struct_iter).first != 0xffffffff) (*struct_iter).second->book( conf_ );
      else (*struct_iter).second->bookAlt( conf_ );
    } 
    else {
      throw cms::Exception("LogicError")
	<< "[SiPixelRawDataErrorSource::bookMEs] Creation of DQM folder failed";
    }

  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelRawDataErrorSource);
