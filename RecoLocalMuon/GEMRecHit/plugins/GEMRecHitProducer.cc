/** \file
 *
 *  \author M. Maggi -- INFN Bari
 */
#include "RecoLocalMuon/GEMRecHit/plugins/GEMRecHitProducer.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"

#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitAlgoFactory.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <fstream>

using namespace edm;
using namespace std;

GEMRecHitProducer::GEMRecHitProducer(const ParameterSet& config)
    : theGEMDigiToken(consumes<GEMDigiCollection>(config.getParameter<InputTag>("gemDigiLabel"))),
      // Get the concrete reconstruction algo from the factory
      theAlgo{GEMRecHitAlgoFactory::get()->create(config.getParameter<string>("recAlgo"),
                                                  config.getParameter<ParameterSet>("recAlgoConfig"))},
      maskSource_(MaskSource::EventSetup),
      deadSource_(MaskSource::EventSetup),
      gemGeomToken_(esConsumes<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun>()) {
  produces<GEMRecHitCollection>();

  // Turns off GE2/1 demonstrator reconstruction in Run3
  ge21Off_ = config.getParameter<bool>("ge21Off");
  // Get masked- and dead-strip information from file
  applyMasking_ = config.getParameter<bool>("applyMasking");
  if (applyMasking_) {
    if (config.existsAs<edm::FileInPath>("maskFile")) {
      maskSource_ = MaskSource::File;
      std::ifstream inputFile(config.getParameter<edm::FileInPath>("maskFile").fullPath());
      if (!inputFile) {
        throw cms::Exception("GEMRecHitProducer") << "Masked Strips File cannot not be opened";
      }
      theGEMMaskedStripsObj = std::make_unique<GEMMaskedStrips>();
      while (inputFile.good()) {
        GEMMaskedStrips::MaskItem Item;
        inputFile >> Item.rawId >> Item.strip;
        if (inputFile.good())
          theGEMMaskedStripsObj->fillMaskVec(Item);
      }
      inputFile.close();
    }

    if (config.existsAs<edm::FileInPath>("deadFile")) {
      deadSource_ = MaskSource::File;
      std::ifstream inputFile(config.getParameter<edm::FileInPath>("deadFile").fullPath());
      if (!inputFile) {
        throw cms::Exception("GEMRecHitProducer") << "Dead Strips File cannot not be opened";
      }
      theGEMDeadStripsObj = std::make_unique<GEMDeadStrips>();
      while (inputFile.good()) {
        GEMDeadStrips::DeadItem Item;
        inputFile >> Item.rawId >> Item.strip;
        if (inputFile.good())
          theGEMDeadStripsObj->fillDeadVec(Item);
      }
      inputFile.close();
    }
    if (maskSource_ == MaskSource::EventSetup) {
      maskedStripsToken_ = esConsumes<GEMMaskedStrips, GEMMaskedStripsRcd, edm::Transition::BeginRun>();
    }
    if (deadSource_ == MaskSource::EventSetup) {
      deadStripsToken_ = esConsumes<GEMDeadStrips, GEMDeadStripsRcd, edm::Transition::BeginRun>();
    }
  }
}

GEMRecHitProducer::~GEMRecHitProducer() = default;

void GEMRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription recAlgoConfigDesc;
  desc.add<edm::ParameterSetDescription>("recAlgoConfig", recAlgoConfigDesc);
  desc.add<std::string>("recAlgo", "GEMRecHitStandardAlgo");
  desc.add<edm::InputTag>("gemDigiLabel", edm::InputTag("muonGEMDigis"));
  desc.add<bool>("applyMasking", false);
  desc.add<bool>("ge21Off", false);
  desc.addOptional<edm::FileInPath>("maskFile");
  desc.addOptional<edm::FileInPath>("deadFile");
  descriptions.add("gemRecHitsDef", desc);
}

void GEMRecHitProducer::beginRun(const edm::Run& r, const edm::EventSetup& setup) {
  // Get the GEM Geometry
  gemGeom_ = setup.getHandle(gemGeomToken_);

  if (applyMasking_) {
    // Getting the masked-strip information
    if (maskSource_ == MaskSource::EventSetup) {
      edm::ESHandle<GEMMaskedStrips> readoutMaskedStrips = setup.getHandle(maskedStripsToken_);
      theGEMMaskedStripsObj = std::make_unique<GEMMaskedStrips>(*readoutMaskedStrips.product());
    }
    // Getting the dead-strip information
    if (deadSource_ == MaskSource::EventSetup) {
      edm::ESHandle<GEMDeadStrips> readoutDeadStrips = setup.getHandle(deadStripsToken_);
      theGEMDeadStripsObj = std::make_unique<GEMDeadStrips>(*readoutDeadStrips.product());
    }

    for (auto gems : gemGeom_->etaPartitions()) {
      // Getting the EtaPartitionMask mask, that includes dead strips, for the given GEMDet
      GEMDetId gemId = gems->id();
      if (ge21Off_ && gemId.station() == 2) {
        continue;
      }
      EtaPartitionMask mask;
      const int rawId = gemId.rawId();
      for (const auto& tomask : theGEMMaskedStripsObj->getMaskVec()) {
        if (tomask.rawId == rawId) {
          const int bit = tomask.strip;
          mask.set(bit);
        }
      }
      for (const auto& tomask : theGEMDeadStripsObj->getDeadVec()) {
        if (tomask.rawId == rawId) {
          const int bit = tomask.strip;
          mask.set(bit);
        }
      }
      // add to masking map if masking present in etaPartition
      if (mask.any()) {
        gemMask_.emplace(gemId, mask);
      }
    }
  }
}

void GEMRecHitProducer::produce(Event& event, const EventSetup& setup) {
  // Get the digis from the event
  Handle<GEMDigiCollection> digis;
  event.getByToken(theGEMDigiToken, digis);

  // Pass the EventSetup to the algo
  theAlgo->setES(setup);

  // Create the pointer to the collection which will store the rechits
  auto recHitCollection = std::make_unique<GEMRecHitCollection>();

  // Iterate through all digi collections ordered by LayerId
  for (auto gemdgIt = digis->begin(); gemdgIt != digis->end(); ++gemdgIt) {
    // The layerId
    const GEMDetId& gemId = (*gemdgIt).first;
    if (ge21Off_ && gemId.station() == 2) {
      continue;
    }

    // Get the GeomDet from the setup
    const GEMEtaPartition* roll = gemGeom_->etaPartition(gemId);
    if (roll == nullptr) {
      edm::LogError("BadDigiInput") << "Failed to find GEMEtaPartition for ID " << gemId;
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

    if (!recHits.empty())  //FIXME: is it really needed?
      recHitCollection->put(gemId, recHits.begin(), recHits.end());
  }

  event.put(std::move(recHitCollection));
}
