/** \file
 *
 *  \author M. Maggi -- INFN Bari
 */

#include "GEMRecHitProducer.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"

#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitAlgoFactory.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

#include "CondFormats/DataRecord/interface/GEMMaskedStripsRcd.h"
#include "CondFormats/DataRecord/interface/GEMDeadStripsRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <fstream>

using namespace edm;
using namespace std;

GEMRecHitProducer::GEMRecHitProducer(const ParameterSet& config):
  theGEMDigiToken(consumes<GEMDigiCollection>(config.getParameter<InputTag>("gemDigiLabel"))),
  // Get the concrete reconstruction algo from the factory
  theAlgo{GEMRecHitAlgoFactory::get()->create(config.getParameter<string>("recAlgo"),
                                              config.getParameter<ParameterSet>("recAlgoConfig"))},
  maskSource_(MaskSource::EventSetup), deadSource_(MaskSource::EventSetup)
{
  produces<GEMRecHitCollection>();

  // Get masked- and dead-strip information from file
  applyMasking_ = config.getParameter<bool>("applyMasking");
  if (applyMasking_) {
    const string maskSource = config.getParameter<std::string>("maskSource");
    if (maskSource == "File") {
      maskSource_ = MaskSource::File;
      edm::FileInPath fp = config.getParameter<edm::FileInPath>("maskvecfile");
      std::ifstream inputFile(fp.fullPath().c_str(), std::ios::in);
      if ( !inputFile ) {
        std::cerr << "Masked Strips File cannot not be opened" << std::endl;
        exit(1);
      }
      theGEMMaskedStripsObj = std::make_unique<GEMMaskedStrips>();
      while ( inputFile.good() ) {
        GEMMaskedStrips::MaskItem Item;
        inputFile >> Item.rawId >> Item.strip;
        if ( inputFile.good() ) theGEMMaskedStripsObj->fillMaskVec(Item);
      }
      inputFile.close();
    }

    const string deadSource = config.getParameter<std::string>("deadSource");
    if (deadSource == "File") {
      deadSource_ = MaskSource::File;
      edm::FileInPath fp = config.getParameter<edm::FileInPath>("deadvecfile");
      std::ifstream inputFile(fp.fullPath().c_str(), std::ios::in);
      if ( !inputFile ) {
        std::cerr << "Dead Strips File cannot not be opened" << std::endl;
        exit(1);
      }
      theGEMDeadStripsObj = std::make_unique<GEMDeadStrips>();
      while ( inputFile.good() ) {
        GEMDeadStrips::DeadItem Item;
        inputFile >> Item.rawId >> Item.strip;
        if ( inputFile.good() ) theGEMDeadStripsObj->fillDeadVec(Item);
      }
      inputFile.close();
    }
  }
}

GEMRecHitProducer::~GEMRecHitProducer() = default;

void GEMRecHitProducer::beginRun(const edm::Run& r, const edm::EventSetup& setup){
  // Get the GEM Geometry
  setup.get<MuonGeometryRecord>().get(gemGeom_);

  if (applyMasking_) {
    // Getting the masked-strip information
    if ( maskSource_ == MaskSource::EventSetup ) {
      edm::ESHandle<GEMMaskedStrips> readoutMaskedStrips;
      setup.get<GEMMaskedStripsRcd>().get(readoutMaskedStrips);
      theGEMMaskedStripsObj = std::make_unique<GEMMaskedStrips>(*readoutMaskedStrips.product());
    }
    // Getting the dead-strip information
    if ( deadSource_ == MaskSource::EventSetup ) {
      edm::ESHandle<GEMDeadStrips> readoutDeadStrips;
      setup.get<GEMDeadStripsRcd>().get(readoutDeadStrips);
      theGEMDeadStripsObj = std::make_unique<GEMDeadStrips>(*readoutDeadStrips.product());
    }

    for (auto gems: gemGeom_->etaPartitions()) {
      // Getting the EtaPartitionMask mask, that includes dead strips, for the given GEMDet
      GEMDetId gemId = gems->id();
      EtaPartitionMask mask;
      const int rawId = gemId.rawId();
      for ( const auto& tomask : theGEMMaskedStripsObj->getMaskVec() ) {
        if ( tomask.rawId == rawId ) {
          const int bit = tomask.strip;
          mask.set(bit-1);
        }
      }
      for ( const auto& tomask : theGEMDeadStripsObj->getDeadVec() ) {
        if ( tomask.rawId == rawId ) {
          const int bit = tomask.strip;
          mask.set(bit-1);
        }
      }
      // add to masking map if masking present in etaPartition
      if (mask.any()) {
        gemMask_.insert(std::pair<GEMDetId,EtaPartitionMask>(gemId, mask));
      }
    }
  }
}

void GEMRecHitProducer::produce(Event& event, const EventSetup& setup)
{
  // Get the digis from the event
  Handle<GEMDigiCollection> digis; 
  event.getByToken(theGEMDigiToken,digis);

  // Pass the EventSetup to the algo
  theAlgo->setES(setup);

  // Create the pointer to the collection which will store the rechits
  auto recHitCollection = std::make_unique<GEMRecHitCollection>();

  // Iterate through all digi collections ordered by LayerId   

  for ( auto gemdgIt = digis->begin(); gemdgIt != digis->end(); ++gemdgIt ) {
    // The layerId
    const GEMDetId& gemId = (*gemdgIt).first;

    // Get the GeomDet from the setup
    const GEMEtaPartition* roll = gemGeom_->etaPartition(gemId);
    if (roll == nullptr){
      edm::LogError("BadDigiInput")<<"Failed to find GEMEtaPartition for ID "<<gemId;
      continue;
    }

    // Get the iterators over the digis associated with this LayerId
    const GEMDigiCollection::Range& range = (*gemdgIt).second;

    // get mask from map
    EtaPartitionMask mask;
    if (applyMasking_) {
      auto gemmaskIt = gemMask_.find(gemId);
      if (gemmaskIt != gemMask_.end())
        mask = gemmaskIt->second;
    }
    
    // Call the reconstruction algorithm    
    OwnVector<GEMRecHit> recHits = theAlgo->reconstruct(*roll, gemId, range, mask);
    
    if(!recHits.empty()) //FIXME: is it really needed?
      recHitCollection->put(gemId, recHits.begin(), recHits.end());
  }

  event.put(std::move(recHitCollection));

}
