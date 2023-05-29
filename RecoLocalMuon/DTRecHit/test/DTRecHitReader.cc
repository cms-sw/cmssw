/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "DTRecHitReader.h"

#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TFile.h"

#include <iostream>
#include <map>

using namespace std;
using namespace edm;

// Constructor
DTRecHitReader::DTRecHitReader(const ParameterSet& pset) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  simHitLabel = pset.getUntrackedParameter<string>("simHitLabel", "r");
  recHitLabel = pset.getUntrackedParameter<string>("recHitLabel", "rechitbuilder");

  if (debug)
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

  dtGeomToken_ = esConsumes();
}

// Destructor
DTRecHitReader::~DTRecHitReader() {
  if (debug)
    cout << "[DTRecHitReader] Destructor called" << endl;

  // Write the histos to file
  theFile->cd();
  hRHitPhi->Write();
  hRHitZ_W0->Write();
  hRHitZ_W1->Write();
  hRHitZ_W2->Write();
  hRHitZ_All->Write();
  theFile->Close();
}

// The real analysis
void DTRecHitReader::analyze(const Event& event, const EventSetup& eventSetup) {
  cout << "--- [DTRecHitReader] Event analysed #Run: " << event.id().run() << " #Event: " << event.id().event() << endl;
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom = eventSetup.getHandle(dtGeomToken_);
  ;

  // Get the rechit collection from the event
  Handle<DTRecHitCollection> dtRecHits;
  event.getByLabel(recHitLabel, dtRecHits);

  // Get the SimHit collection from the event
  Handle<PSimHitContainer> simHits;

  event.getByLabel(simHitLabel, "MuonDTHits", simHits);

  if (debug)
    cout << "   #SimHits: " << simHits->size() << endl;

  // Map simhits per wire
  map<DTWireId, vector<const PSimHit*> > simHitMap = mapSimHitsPerWire(simHits);

  // Iterate over all detunits
  DTRecHitCollection::id_iterator detUnitIt;
  for (detUnitIt = dtRecHits->id_begin(); detUnitIt != dtRecHits->id_end(); ++detUnitIt) {
    // Get the GeomDet from the setup
    const DTLayer* layer = dtGeom->layer(*detUnitIt);

    // Get the range for the corresponding LayerId
    DTRecHitCollection::range range = dtRecHits->get((*detUnitIt));
    // Loop over the rechits of this DetUnit
    for (DTRecHitCollection::const_iterator rechit = range.first; rechit != range.second; ++rechit) {
      // Get the wireId of the rechit
      DTWireId wireId = (*rechit).wireId();

      float xwire = layer->specificTopology().wirePosition(wireId.wire());
      if (fabs(xwire - (*rechit).localPosition().x()) > 0.00001) {
        cout << "  [DTRecHitReader]***Error in wire Position: xwire = " << xwire
             << " xRecHit = " << (*rechit).localPosition().x() << endl;
      }

      // Access to Right and left rechits
      pair<const DTRecHit1D*, const DTRecHit1D*> lrRecHits = (*rechit).componentRecHits();

      cout << "Left Hit x(cm): " << lrRecHits.first->localPosition().x() << endl;
      cout << "Right Hit x(cm): " << lrRecHits.second->localPosition().x() << endl;

      // Compute the rechit distance from wire
      float distFromWire =
          fabs((*rechit).localPosition(DTEnums::Left).x() - (*rechit).localPosition(DTEnums::Right).x()) / 2.;

      // Search the best mu simhit and compute its distance from the wire
      float simHitDistFromWire = 0;
      if (simHitMap.find(wireId) != simHitMap.end()) {
        const PSimHit* muSimHit = findBestMuSimHit(layer, wireId, simHitMap[wireId], distFromWire);
        // Check that a mu simhit is found
        if (muSimHit != 0) {
          // Compute the simhit distance from wire
          simHitDistFromWire = findSimHitDist(layer, wireId, muSimHit);
          // Fill the histos
          H1DRecHit* histo = 0;
          if (wireId.superlayer() == 1 || wireId.superlayer() == 3) {
            histo = hRHitPhi;
          } else if (wireId.superlayer() == 2) {
            hRHitZ_All->Fill(distFromWire, simHitDistFromWire);
            if (wireId.wheel() == 0) {
              histo = hRHitZ_W0;
            } else if (abs(wireId.wheel()) == 1) {
              histo = hRHitZ_W1;
            } else if (abs(wireId.wheel()) == 2) {
              histo = hRHitZ_W2;
            }
          }
          histo->Fill(distFromWire, simHitDistFromWire);

          if (fabs(distFromWire - simHitDistFromWire) > 2.1) {
            cout << "Warning: " << endl
                 << "  RecHit distance from wire is: " << distFromWire << endl
                 << "  SimHit distance from wire is: " << simHitDistFromWire << endl
                 << "  RecHit wire Id is: " << wireId << endl
                 << "  SimHit wire Id is: " << DTWireId((*muSimHit).detUnitId()) << endl;
            cout << "  Wire x: = " << xwire << endl
                 << "  RecHit x = " << (*rechit).localPosition().x() << endl
                 << "  SimHit x = " << (*muSimHit).localPosition().x() << endl;
          }

          // Some printout
          if (debug) {
            cout << "[DTRecHitReader]: " << endl
                 << "         WireId: " << wireId << endl
                 << "         1DRecHitPair local position (cm): " << (*rechit) << endl
                 << "         RecHit distance from wire (cm): " << distFromWire << endl
                 << "         Mu SimHit distance from wire (cm): " << simHitDistFromWire << endl;
          }
        }
      }
    }
  }
}

// Return a map between simhits of a layer and the wireId of their cell
map<DTWireId, vector<const PSimHit*> > DTRecHitReader::mapSimHitsPerWire(const Handle<PSimHitContainer>& simhits) {
  map<DTWireId, vector<const PSimHit*> > hitWireMapResult;

  for (PSimHitContainer::const_iterator simhit = simhits->begin(); simhit != simhits->end(); simhit++) {
    hitWireMapResult[DTWireId((*simhit).detUnitId())].push_back(&(*simhit));
  }

  return hitWireMapResult;
}

const PSimHit* DTRecHitReader::findBestMuSimHit(const DTLayer* layer,
                                                const DTWireId& wireId,
                                                const vector<const PSimHit*>& simhits,
                                                float recHitDistFromWire) {
  const PSimHit* retSimHit = 0;
  float tmp_distDiff = 999999;
  for (vector<const PSimHit*>::const_iterator simhit = simhits.begin(); simhit != simhits.end(); simhit++) {
    // Select muons
    if (abs((*simhit)->particleType()) == 13) {
      // Get the mu simhit closest to the rechit
      if (findSimHitDist(layer, wireId, *simhit) - recHitDistFromWire < tmp_distDiff) {
        tmp_distDiff = findSimHitDist(layer, wireId, *simhit) - recHitDistFromWire;
        retSimHit = (*simhit);
      }
    }
  }
  return retSimHit;
}

// Compute SimHit distance from wire
double DTRecHitReader::findSimHitDist(const DTLayer* layer, const DTWireId& wireId, const PSimHit* hit) {
  float xwire = layer->specificTopology().wirePosition(wireId.wire());
  LocalPoint entryP = hit->entryPoint();
  LocalPoint exitP = hit->exitPoint();
  float xEntry = entryP.x() - xwire;
  float xExit = exitP.x() - xwire;

  return fabs(xEntry - (entryP.z() * (xExit - xEntry)) / (exitP.z() - entryP.z()));  //FIXME: check...
}

DEFINE_FWK_MODULE(DTRecHitReader);
