/**
 *  See header file for a description of this class.
 *  
 *  \author Dominique Fortin - UCR
 */

#include "RecoMuon/MuonSeedGenerator/plugins/MuonSeedProducer.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedBuilder.h"

// Data Formats
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <DataFormats/Common/interface/Handle.h>
// C++
#include <vector>

/* 
 * Constructor
 */
MuonSeedProducer::MuonSeedProducer(const edm::ParameterSet& pset) {
  // Register what this produces
  produces<TrajectorySeedCollection>();

  // Local Debug flag
  debug = pset.getParameter<bool>("DebugMuonSeed");

  edm::ConsumesCollector iC = consumesCollector();

  // Builder which returns seed collection
  muonSeedBuilder_ = new MuonSeedBuilder(pset, iC);

  muonLayersToken_ = esConsumes<MuonDetLayerGeometry, MuonRecoGeometryRecord>();
  magFieldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
}

/*
 * Destructor
 */
MuonSeedProducer::~MuonSeedProducer() { delete muonSeedBuilder_; }

/*
 * Producer (the main)
 */
void MuonSeedProducer::produce(edm::Event& event, const edm::EventSetup& eSetup) {
  // Muon Geometry
  edm::ESHandle<MuonDetLayerGeometry> muonLayers = eSetup.getHandle(muonLayersToken_);
  const MuonDetLayerGeometry* lgeom = &*muonLayers;
  muonSeedBuilder_->setGeometry(lgeom);

  // Magnetic field
  edm::ESHandle<MagneticField> field = eSetup.getHandle(magFieldToken_);
  const MagneticField* theField = &*field;
  muonSeedBuilder_->setBField(theField);

  // Create pointer to the seed container

  auto output = std::make_unique<TrajectorySeedCollection>();

  //UNUED:  int nSeeds = 0;
  //UNUSED: nSeeds =
  muonSeedBuilder_->build(event, eSetup, *output);

  // Append muon seed collection to event
  event.put(std::move(output));
}
