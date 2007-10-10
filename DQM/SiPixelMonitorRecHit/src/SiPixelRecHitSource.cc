// -*- C++ -*-
//
// Package:    SiPixelMonitorRecHits
// Class:      SiPixelRecHitSource
// 
/**\class 

 Description: Pixel DQM source for RecHits

 Implementation:
     Originally based on the code for Digis, adapted
	to read RecHits and create relevant histograms
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id: SiPixelDigiSource.cc,v 1.15 2007/09/04 18:21:30 merkelp Exp $
//
//
// Adapted by:  Keith Rose
//  	For use in SiPixelMonitorClient for RecHits

#include "DQM/SiPixelMonitorRecHit/interface/SiPixelRecHitSource.h"
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
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
// SimHit stuff
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
//
#include <boost/cstdint.hpp>
#include <string>
#include <stdlib.h>
#include <iostream>
using namespace std;
using namespace edm;

SiPixelRecHitSource::SiPixelRecHitSource(const edm::ParameterSet& iConfig) :
  conf_(iConfig),
  src_( conf_.getParameter<edm::InputTag>( "src" ) )
{
   theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
   LogInfo ("PixelDQM") << "SiPixelRecHitSource::SiPixelRecHitSource: Got DQM BackEnd interface"<<endl;
}


SiPixelRecHitSource::~SiPixelRecHitSource()
{
   // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelRecHitSource::~SiPixelRecHitSource: Destructor"<<endl;
}


void SiPixelRecHitSource::beginJob(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") << " SiPixelRecHitSource::beginJob - Initialisation ... " << std::endl;
  eventNo = 0;
	cout << "0" << endl;
  // Build map
  buildStructure(iSetup);
	cout << "1" << endl;
  // Book Monitoring Elements
  bookMEs();

}


void SiPixelRecHitSource::endJob(void){
  cout << " SiPixelDigiSource::endJob - Saving Root File " << std::endl;
  std::string outputFile = conf_.getParameter<std::string>("outputFile");
  cout << "ending" << endl;
  theDMBE->save( outputFile.c_str() );
  cout << "last" << endl;
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelRecHitSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;
	cout << eventNo << endl;
  // get input data
  edm::Handle<SiPixelRecHitCollection>  recHitColl;
  iEvent.getByLabel( src_, recHitColl );

  TrackerHitAssociator associate(iEvent, conf_ );
  std::map<uint32_t,SiPixelRecHitModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    uint32_t TheID = (*struct_iter).first;
	
    SiPixelRecHitCollection::range pixelrechitRange = (recHitColl.product())->get(TheID);
    SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.first;
    
	SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.second;
      SiPixelRecHitCollection::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
      std::vector<PSimHit> matched;
	if( pixelrechitRangeIteratorBegin == pixelrechitRangeIteratorEnd) {cout << "oops" << endl;}
      float x_res = 0;
      float y_res = 0;
      float rechit_x = 0;
      float rechit_y = 0;
      float x_pull = 0;
      float y_pull = 0;

  for ( ; pixeliter != pixelrechitRangeIteratorEnd; pixeliter++) 
	{
	  matched.clear();
	  matched = associate.associateHit(*pixeliter);
	cout << "dd" << endl;	  
	  if ( !matched.empty() ) 
	    {
		cout << "ee" << endl;
	      float closest = 9999.9;
	      std::vector<PSimHit>::const_iterator closestit = matched.begin();
	      LocalPoint lp = pixeliter->localPosition();
	      rechit_x = lp.x();
	      rechit_y = lp.y();
	      LocalError lerr = pixeliter->localPositionError();
	      float lerr_x = sqrt(lerr.xx());
	      float lerr_y = sqrt(lerr.yy());
      //loop over sim hits and fill closet
	      for (std::vector<PSimHit>::const_iterator m = matched.begin(); m<matched.end(); m++) 
		{
	cout << "ff" << endl;
		  float sim_x1 = (*m).entryPoint().x();
		  float sim_x2 = (*m).exitPoint().x();
		  float sim_xpos = 0.5*(sim_x1+sim_x2);

		  float sim_y1 = (*m).entryPoint().y();
		  float sim_y2 = (*m).exitPoint().y();
		  float sim_ypos = 0.5*(sim_y1+sim_y2);
		  
		  float x_resa = fabs(sim_xpos - rechit_x);
		  float y_resa = fabs(sim_ypos - rechit_y);

		  
		  
		  float dist = sqrt(x_resa*x_resa + y_resa*y_resa);

		  if ( dist < closest ) 
		    {
		      closest = x_resa;
		      closestit = m;
		    }
		} // end sim hit loop
		
		cout << (*closestit).entryPoint().x() << endl;
	      float sim_x1 = (*closestit).entryPoint().x();
	      float sim_x2 = (*closestit).exitPoint().x();
	      float sim_y1 = (*closestit).entryPoint().y();
	      float sim_y2 = (*closestit).exitPoint().y();
	      float sim_xpos = .5*(sim_x1+sim_x2);
	      float sim_ypos = .5*(sim_y1+sim_y2);
	      x_res = (rechit_x - sim_xpos) * 10000;
		cout << x_res << endl;
	      y_res = (rechit_y - sim_ypos) * 10000;
	      x_pull = (rechit_x - sim_xpos) / lerr_x;
	      y_pull = (rechit_y - sim_ypos) / lerr_y;
	      (*struct_iter).second->fill(rechit_x, rechit_y, x_res, y_res, x_pull, y_pull);
	    }
	}
    
    
  }

  // slow down...
  //usleep(100000);
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelRecHitSource::buildStructure(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") <<" SiPixelRecHitSource::buildStructure" ;
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

     	  
	  
	      // SiPixelRecHitModule *theModule = new SiPixelRecHitModule(id, rechit_x, rechit_y, x_res, y_res, x_pull, y_pull);
	
	
	      if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
		LogDebug ("PixelDQM") << " ---> Adding Barrel Module " <<  detId.rawId() << endl;
		uint32_t id = detId();
	
	SiPixelRecHitModule* theModule = new SiPixelRecHitModule(id);
		thePixelStructure.insert(pair<uint32_t,SiPixelRecHitModule*> (id,theModule));
		
	      }	else if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
		LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
		uint32_t id = detId();
		SiPixelRecHitModule* theModule = new SiPixelRecHitModule(id);
		thePixelStructure.insert(pair<uint32_t,SiPixelRecHitModule*> (id,theModule));
	      }
      
	}	    
  }

  LogInfo ("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelRecHitSource::bookMEs(){
  
  std::map<uint32_t,SiPixelRecHitModule*>::iterator struct_iter;
  theDMBE->setVerbose(0);
    
  SiPixelFolderOrganizer theSiPixelFolder;
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    
    /// Create folder tree and book histograms 
    if(theSiPixelFolder.setModuleFolder((*struct_iter).first)){
      (*struct_iter).second->book( conf_ );
    } else {
      throw cms::Exception("LogicError")
	<< "[SiPixelDigiSource::bookMEs] Creation of DQM folder failed";
    }
    
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelRecHitSource);
