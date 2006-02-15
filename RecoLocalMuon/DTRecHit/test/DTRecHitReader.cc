/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTRecHitReader.h"


#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "TFile.h"

#include <iostream>
#include <map>



using namespace std;
using namespace edm;


// Constructor
DTRecHitReader::DTRecHitReader(const ParameterSet& pset){
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  rootFileName = pset.getUntrackedParameter<string>("rootFileName");

  if(debug)
    cout << "[DTRecHitReader] Constructor called" << endl;

  // Create the root file
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  // Book the histograms
  hRHitPhi = new H1DRecHit("RPhi");
  hRHitZ_W0 = new H1DRecHit("RZ_W0");
  hRHitZ_W1 = new H1DRecHit("RZ_W1");
  hRHitZ_W2 = new H1DRecHit("RZ_W2");
  hRHitZ_All = new H1DRecHit("RZ_All");
}



// Destructor
DTRecHitReader::~DTRecHitReader(){
  if(debug) 
    cout << "[DTRecHitReader] Destructor called" << endl;

  // Write the histos to file
  theFile->cd();
  hRHitPhi->Write();
  hRHitZ_W0->Write();
  hRHitZ_W1->Write();
  hRHitZ_W2->Write();
  hRHitZ_All->Write();
  theFile->Close();
  //delete hRHitPhi; //FIXME: This one makes a mess
}



// The real analysis
void DTRecHitReader::analyze(const Event & event, const EventSetup& eventSetup){
  cout << "--- [DTRecHitReader] Event analysed #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;

  // Get the rechit collection from the event
  Handle<DTRecHitCollection> dtTecHits;
  event.getByLabel("rechitbuilder", dtTecHits);

  // Get the SimHit collection from the event
  Handle<PSimHitContainer> simHits;
  event.getByLabel("r","MuonDTHits", simHits);
  
  if(debug)
    cout << "   #SimHits: " << simHits->size() << endl;

  // Map simhits per wire
  map<DTWireId, vector<const PSimHit*> > simHitMap =
    mapSimHitsPerWire(simHits);

  // Iterate over all detunits
  DTRecHitCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = dtTecHits->begin();
       detUnitIt != dtTecHits->end();
       ++detUnitIt){
    //     const DTLayerId& layerId = (*detUnitIt).first;
    const DTRecHitCollection::Range& range = (*detUnitIt).second;
      
    // Loop over the rechits of this DetUnit
    for (DTRecHitCollection::const_iterator rechit = range.first;
	 rechit!=range.second;
	   ++rechit){
      // Get the wireId of the rechit
      DTWireId wireId = (*rechit).wireId();

      // Compute the rechit distance from wire
      float distFromWire = fabs((*rechit).localPosition(DTEnums::Left).x() -
				(*rechit).localPosition(DTEnums::Right).x())/2.;

      // Search the best mu simhit and compute its distance from the wire
      float simHitDistFromWire = 0;
      if(simHitMap.find(wireId) != simHitMap.end()) {
	const PSimHit* muSimHit = findBestMuSimHit(simHitMap[wireId], distFromWire);
	// Check that a mu simhit is found
	if(muSimHit != 0) {
	  // Compute the simhit distance from wire
	  simHitDistFromWire = fabs(muSimHit->localPosition().x());
	  // Fill the histos
	  H1DRecHit *histo = 0;
	  if(wireId.superlayer() == 1 || wireId.superlayer() == 3) {
	    histo = hRHitPhi;
	  } else if(wireId.superlayer() == 2) {
	    hRHitZ_All->Fill(distFromWire, simHitDistFromWire);
	    if(wireId.wheel() == 0) {
	      histo = hRHitZ_W0;
	    } else if(abs(wireId.wheel()) == 1) {
	      histo = hRHitZ_W1;
	    } else if(abs(wireId.wheel()) == 2) {
	      histo = hRHitZ_W2;
	    }
	  }
	  histo->Fill(distFromWire, simHitDistFromWire);
	  
	  // Some printout
	  if(debug) {
	    cout << "[DTRecHitReader]: " << endl
		 << "         WireId: " << wireId << endl
		 << "         1DRecHitPair local position (cm): " << (*rechit) << endl
		 << "         RecHit distance from wire (cm): " << distFromWire << endl
		 << "         Mu SimHit sidtance from wire (cm): " << simHitDistFromWire << endl;
	  }
	}
      }
    }
  }
}



// Return a map between simhits of a layer and the wireId of their cell
map<DTWireId, vector<const PSimHit*> >
DTRecHitReader::mapSimHitsPerWire(const Handle<PSimHitContainer>& simhits) {
   map<DTWireId, vector<const PSimHit*> > hitWireMapResult;
   
   for(PSimHitContainer::const_iterator simhit = simhits->begin();
       simhit != simhits->end();
       simhit++) {
     hitWireMapResult[DTWireId((*simhit).detUnitId())].push_back(&(*simhit));
   }
   
   return hitWireMapResult;
}


const PSimHit*
DTRecHitReader::findBestMuSimHit(const vector<const PSimHit*>& simhits,
				 float recHitDistFromWire) {
  const PSimHit* retSimHit =0;
  float tmp_distDiff = 999999;
  for(vector<const PSimHit*>::const_iterator simhit = simhits.begin();
      simhit != simhits.end();
      simhit++) {
    // Select muons
    if(abs((*simhit)->particleType()) == 13) {
      // Get the mu simhit closest to the rechit
      if(fabs((*simhit)->localPosition().x())-recHitDistFromWire < tmp_distDiff) {
	tmp_distDiff = fabs((*simhit)->localPosition().x())-recHitDistFromWire;
	retSimHit = (*simhit);
      }
    }
  }
  return retSimHit;
}



DEFINE_FWK_MODULE(DTRecHitReader)

