/*
 *  Basic analyzer class which accesses CSCRecHits
 *  and compare them with muon simhits and/or digis.  
 *
 *  The output histograms are dealt with in CSCRecHitHistograms.h
 *
 *  Author: R. Trentadue - Bari University
 */

#include "RPCRecHitReader.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/RPCGeometry/interface/RPCRoll.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
 
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TFile.h"
#include "TVector3.h"

#include <iostream>
#include <map>

using namespace std;
using namespace edm;


// Constructor
RPCRecHitReader::RPCRecHitReader(const ParameterSet& pset){

  // Get the various input parameters
  simHitLabel1      = pset.getUntrackedParameter<string>("simHitLabel1");
  simHitLabel2      = pset.getUntrackedParameter<string>("simHitLabel2");
  digiLabel         = pset.getUntrackedParameter<string>("digiLabel");
  recHitLabel1      = pset.getUntrackedParameter<string>("recHitLabel1");
  recHitLabel2      = pset.getUntrackedParameter<string>("recHitLabel2");
}

// Destructor
RPCRecHitReader::~RPCRecHitReader(){}

// The Analysis  (the main)
void RPCRecHitReader::analyze(const Event & event, const EventSetup& eventSetup){
  
  if (event.id().event()%100 == 0) cout << " Event analysed #Run: " << event.id().run()
					<< " #Event: " << event.id().event() << endl;
  
  // Get the RPC Geometry :
  ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);
  
  // Get the SimHit collection :
//   Handle<PSimHitContainer> simHits;
//   event.getByLabel(simHitLabel1, simHitLabel2, simHits); 

//   cout << "   #SimHits: " << simHits->size() << endl;

  // Get the Digis collection:
//   Handle<RPCDigiCollection> rpcDigis;
//   event.getByLabel(digiLabel, rpcDigis);

  // Get the RecHits collection :
  Handle<RPCRecHitCollection> recHits; 
  event.getByLabel(recHitLabel1, recHitLabel2, recHits);  
  cout << "   #RecHits: " << recHits->size() << endl;
  
  // First loop over simhits and count how many simhits you have per chambers 
  
//   if (simHits->size() > 200) {
//     cout << "More simhits than allowed: " << simHits->size() << " > 200  --> skip event" << endl;
//     return;
//   }
  
  // Build iterator for simHits:
//   PSimHitContainer::const_iterator simIt_1;
  
//   // Search for matching hit in layer/chamber/...  in simhits:
//   for (simIt_1 = simHits->begin(); simIt_1 != simHits->end(); simIt_1++) {
    
//     // Find chamber where simhit is located
//     RPCDetId id_1 = (RPCDetId)(*simIt_1).detUnitId();
    
//     std::cout<<"Region    = "<< id_1.region()    << std::endl;
//     std::cout<<"Ring      = "<< id_1.ring()      << std::endl;
//     std::cout<<"Station   = "<< id_1.station()   << std::endl;
//     std::cout<<"Sector    = "<< id_1.sector()    << std::endl;
//   }
  
  // Loop over rechits 
  
  // Build iterator for rechits and loop :
  RPCRecHitCollection::const_iterator recIt;
      
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
	
    // Find chamber with rechits in RPC 
    RPCDetId idrec = (RPCDetId)(*recIt).rpcId();
	
//     std::cout<<"Region    = "<< idrec.region()    << std::endl;
//     std::cout<<"Ring      = "<< idrec.ring()      << std::endl;
//     std::cout<<"Station   = "<< idrec.station()   << std::endl;
//     std::cout<<"Sector    = "<< idrec.sector()    << std::endl;

    LocalPoint rhitlocal_temp = (*recIt).localPosition(); 
    LocalError rhitlocalerr_temp = (*recIt).localPositionError(); 
     
    std::cout<<"Local Position = "<<rhitlocal_temp<<"  "<<"Local error "<<rhitlocalerr_temp<<std::endl;
    std::cout<<"First Strip    = "<<(*recIt).firstClusterStrip()<<std::endl;
    std::cout<<"Cluster Size   = "<<(*recIt).clusterSize()<<std::endl;

//     std::cout<<"Local X = "<<rhitlocal_temp.x()<<"  "<<"Local error X"<<rhitlocalerr_temp.x()<<std::endl;
//     std::cout<<"Local Y = "<<rhitlocal_temp.y()<<"  "<<"Local error Y"<<rhitlocalerr_temp.y()<<std::endl;
//     std::cout<<"Local Z = "<<rhitlocal_temp.z()<<"  "<<"Local error Z"<<rhitlocalerr_temp.z()<<std::endl;

//     // Find out the corresponding strip #
//     stripnum = geom->nearestStrip(rhitlocal);
  }
}

DEFINE_FWK_MODULE(RPCRecHitReader)
