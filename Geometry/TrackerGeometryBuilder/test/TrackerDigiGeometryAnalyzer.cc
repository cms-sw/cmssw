// -*- C++ -*-
//
// Package:    TrackerDigiGeometryAnalyzer
// Class:      TrackerDigiGeometryAnalyzer
// 
/**\class TrackerDigiGeometryAnalyzer TrackerDigiGeometryAnalyzer.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filippo Ambroglini
//         Created:  Tue Jul 26 08:47:57 CEST 2005
// $Id: TrackerDigiGeometryAnalyzer.cc,v 1.7 2005/11/04 18:18:54 fambrogl Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerTopology/interface/PixelTopology.h"
#include "Geometry/TrackerTopology/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Surface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
//
// class decleration
//

class TrackerDigiGeometryAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TrackerDigiGeometryAnalyzer( const edm::ParameterSet& );
      ~TrackerDigiGeometryAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackerDigiGeometryAnalyzer::TrackerDigiGeometryAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed

}


TrackerDigiGeometryAnalyzer::~TrackerDigiGeometryAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackerDigiGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   edm::LogInfo("TrackerDigiGeometryAnalyzer")<< "Here I am";
   //
   // get the TrackerGeom
   //
   edm::ESHandle<TrackerGeometry> pDD;
   iSetup.get<TrackerDigiGeometryRecord>().get( pDD );     
   edm::LogInfo("TrackerDigiGeometryAnalyzer")<< " Geometry node for TrackerGeom is  "<<&(*pDD);   
   edm::LogInfo("TrackerDigiGeometryAnalyzer")<<" I have "<<pDD->dets().size() <<" detectors";
   edm::LogInfo("TrackerDigiGeometryAnalyzer")<<" I have "<<pDD->detTypes().size() <<" types";

   for(TrackingGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
       if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
	BoundSurface& p = (dynamic_cast<PixelGeomDetUnit*>((*it)))->surface();
        edm::LogInfo("TrackerDigiGeometryAnalyzer")<<"    Thickness "<<p.bounds().thickness();
       } 
    }	

   for (TrackingGeometry::DetTypeContainer::const_iterator it = pDD->detTypes().begin(); it != pDD->detTypes().end(); it ++){
     if (dynamic_cast<PixelGeomDetType*>((*it))!=0){
       edm::LogInfo("TrackerDigiGeometryAnalyzer")<<" PIXEL Det";
       PixelTopology& p = (dynamic_cast<PixelGeomDetType*>((*it)))->specificTopology();
       edm::LogInfo("TrackerDigiGeometryAnalyzer")<<"    Rows    "<<p.nrows();
       edm::LogInfo("TrackerDigiGeometryAnalyzer")<<"    Columns "<<p.ncolumns();
   }else{
       edm::LogInfo("TrackerDigiGeometryAnalyzer") <<" STRIP Det";
       StripTopology& p = (dynamic_cast<StripGeomDetType*>((*it)))->specificTopology();
       edm::LogInfo("TrackerDigiGeometryAnalyzer")<<"    Strips    "<<p.nstrips();
     }
     
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerDigiGeometryAnalyzer)
