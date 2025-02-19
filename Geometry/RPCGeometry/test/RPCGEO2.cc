// -*- C++ -*-
//
// Package:    RPCGEO2
// Class:      RPCGEO2
// 
/**\class RPCGEO2 RPCGEO2.cc rpcgeo/RPCGEO2/src/RPCGEO2.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/91
//         Created:  Wed Sep 26 17:08:29 CEST 2007
// $Id: RPCGEO2.cc,v 1.2 2011/11/30 15:20:52 mmaggi Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>

//
// class decleration
//

class RPCGEO2 : public edm::EDAnalyzer {
   public:
      explicit RPCGEO2(const edm::ParameterSet&);
      ~RPCGEO2();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

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
RPCGEO2::RPCGEO2(const edm::ParameterSet& /*iConfig*/){
   //now do what ever initialization is needed
}


RPCGEO2::~RPCGEO2()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
RPCGEO2::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup)
{
   using namespace edm;

   std::cout <<" Getting the RPC Geometry"<<std::endl;
   edm::ESHandle<RPCGeometry> rpcGeo;
   iSetup.get<MuonGeometryRecord>().get(rpcGeo);

   for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
     if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
       RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
       std::vector< const RPCRoll*> roles = (ch->rolls());
       
       //std::cout<<"RPC Chamber"<<ch->id()<<std::endl;
       
       for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	 RPCDetId rpcId = (*r)->id();
	 //int stripsinthisroll=(*r)->nstrips();
	 RPCGeomServ rpcsrv(rpcId);
	 if (rpcId.region()==0){ 
	   //	   const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&((*r)->topology()));
	   //	   float stripl = top_->stripLength();
	   //	   float stripw = top_->pitch();
	   const BoundPlane & RPCSurface = (*r)->surface();
	   GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
	   std::cout<<rpcsrv.name()<<" "<<CenterPointRollGlobal.x()<<" "<<CenterPointRollGlobal.y()<<" "<<CenterPointRollGlobal.z()<<std::endl;
	   GlobalPoint i = RPCSurface.toGlobal(LocalPoint(1,0,0));
	   GlobalPoint j = RPCSurface.toGlobal(LocalPoint(0,1,0));
	   std::cout<<" i "<<i.x()<<" "<<i.y()<<" "<<i.z()<<std::endl;
	   std::cout<<" j "<<j.x()<<" "<<j.y()<<" "<<j.z()<<std::endl;

	   

	 }else{
	   //	   const TrapezoidalStripTopology* top_= dynamic_cast<const TrapezoidalStripTopology*> (&((*r)->topology()));
	   //	   float stripl = top_->stripLength();
	   //float stripw = top_->pitch();
	   const BoundPlane & RPCSurface = (*r)->surface();
	   GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
	   std::cout<<rpcsrv.name()<<" "<<CenterPointRollGlobal.x()<<" "<<CenterPointRollGlobal.y()<<" "<<CenterPointRollGlobal.z()<<std::endl;
	 }
       }
     }
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
RPCGEO2::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCGEO2::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCGEO2);
