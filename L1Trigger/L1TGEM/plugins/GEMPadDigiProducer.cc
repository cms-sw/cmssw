#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include <set>

/// \class GEMPadDigiProducer
/// producer for GEM trigger pads

class GEMPadDigiProducer : public edm::stream::EDProducer<> {
public:
  explicit GEMPadDigiProducer(const edm::ParameterSet& ps);

  ~GEMPadDigiProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void buildPads(const GEMDigiCollection& digis, GEMPadDigiCollection& out_pads) const;
  void buildPads16GE21(const GEMDigiCollection& digis, GEMPadDigiCollection& out_pads) const;
  void checkValid(const GEMPadDigi& pad, const GEMDetId& id) const;
  void checkGeometry() const;

  /// Name of input digi Collection
  edm::EDGetTokenT<GEMDigiCollection> digi_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geom_token_;
  edm::InputTag digis_;
  bool use16GE21_;

  const GEMGeometry* geometry_;
};

GEMPadDigiProducer::GEMPadDigiProducer(const edm::ParameterSet& ps) : geometry_(nullptr) {
  digis_ = ps.getParameter<edm::InputTag>("InputCollection");

  digi_token_ = consumes<GEMDigiCollection>(digis_);
  geom_token_ = esConsumes<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();

  produces<GEMPadDigiCollection>();
  consumes<GEMDigiCollection>(digis_);
}

GEMPadDigiProducer::~GEMPadDigiProducer() {}

void GEMPadDigiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputCollection", edm::InputTag("simMuonGEMDigis"));

  descriptions.add("simMuonGEMPadDigisDef", desc);
}

void GEMPadDigiProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  edm::ESHandle<GEMGeometry> hGeom = eventSetup.getHandle(geom_token_);
  geometry_ = &*hGeom;
  // check the number of parititions
  if (geometry_->hasGE21()) {
    use16GE21_ = (geometry_->chamber(GEMDetId(1, 1, 2, 1, 1, 0))->nEtaPartitions() ==
                  GEMPadDigi::NumberPartitions::GE21SplitStrip);
  }

  checkGeometry();
}

void GEMPadDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Handle<GEMDigiCollection> hdigis;
  e.getByToken(digi_token_, hdigis);

  // Create empty output
  std::unique_ptr<GEMPadDigiCollection> pPads(new GEMPadDigiCollection());

  // build the pads
  buildPads(*(hdigis.product()), *pPads);
  if (use16GE21_)
    buildPads16GE21(*(hdigis.product()), *pPads);

  // store them in the event
  e.put(std::move(pPads));
}

void GEMPadDigiProducer::buildPads(const GEMDigiCollection& det_digis, GEMPadDigiCollection& out_pads) const {
  for (const auto& p : geometry_->etaPartitions()) {
    // when using the GE2/1 geometry with 16 eta partitions
    // ->ignore GE2/1
    if (use16GE21_ and p->isGE21())
      continue;

    // set of <pad, bx, part> pairs, sorted first by pad then by bx
    std::set<std::tuple<int, int, unsigned> > proto_pads;

    // walk over digis in this partition,
    // and stuff them into a set of unique pads (equivalent of OR operation)
    auto digis = det_digis.get(p->id());
    for (auto d = digis.first; d != digis.second; ++d) {
      unsigned pad_num = static_cast<unsigned>(p->padOfStrip(d->strip()));

      auto nPart = GEMPadDigi::NumberPartitions::GE11;
      if (p->isME0()) {
        nPart = GEMPadDigi::NumberPartitions::ME0;
      } else if (p->isGE21()) {
        nPart = GEMPadDigi::NumberPartitions::GE21;
      }
      // check that the input digi is valid
      if ((p->isGE11() and pad_num == GEMPadDigi::GE11InValid) or
          (p->isGE21() and pad_num == GEMPadDigi::GE21InValid) or (p->isME0() and pad_num == GEMPadDigi::ME0InValid)) {
        edm::LogWarning("GEMPadDigiProducer") << "Invalid " << pad_num << " from  " << *d << " in " << p->id();
      }
      proto_pads.emplace(pad_num, d->bx(), nPart);
    }

    // fill the output collections
    for (const auto& d : proto_pads) {
      GEMPadDigi pad_digi(std::get<0>(d), std::get<1>(d), p->subsystem(), std::get<2>(d));
      checkValid(pad_digi, p->id());
      out_pads.insertDigi(p->id(), pad_digi);
    }
  }
}

void GEMPadDigiProducer::buildPads16GE21(const GEMDigiCollection& det_digis, GEMPadDigiCollection& out_pads) const {
  for (const auto& p : geometry_->etaPartitions()) {
    // when using the GE2/1 geometry with 16 eta partitions
    // ->ignore GE1/1
    if (!p->isGE21())
      continue;

    // ignore eta partition with even numbers
    // these are included in the odd numbered pads
    if (p->id().roll() % 2 == 0)
      continue;

    // set of <pad, bx> pairs, sorted first by pad then by bx
    std::set<std::pair<int, int> > proto_pads;

    // walk over digis in the first partition,
    // and stuff them into a set of unique pads (equivalent of OR operation)
    auto digis = det_digis.get(p->id());

    GEMDetId gemId2(
        p->id().region(), p->id().ring(), p->id().station(), p->id().layer(), p->id().chamber(), p->id().roll() + 1);
    auto digis2 = det_digis.get(gemId2);

    for (auto d = digis.first; d != digis.second; ++d) {
      // check if the strip digi in the eta partition below also has a digi
      for (auto d2 = digis2.first; d2 != digis2.second; ++d2) {
        if (d->strip() == d2->strip()) {
          proto_pads.emplace(d->strip(), d->bx());
        }
      }
    }

    // fill the output collections
    for (const auto& d : proto_pads) {
      GEMPadDigi pad_digi(d.first, d.second, p->subsystem(), GEMPadDigi::NumberPartitions::GE21SplitStrip);
      checkValid(pad_digi, p->id());
      out_pads.insertDigi(p->id(), pad_digi);
    }
  }
}

void GEMPadDigiProducer::checkValid(const GEMPadDigi& pad, const GEMDetId& id) const {
  // check if the pad is valid
  // in principle, invalid pads can appear in the CMS raw data
  if (!pad.isValid()) {
    edm::LogWarning("GEMPadDigiProducer") << "Invalid " << pad << " in " << id;
  }
}

void GEMPadDigiProducer::checkGeometry() const {
  // check that ME0 has 8-eta partitions
  if (geometry_->hasME0()) {
    if (geometry_->chamber(GEMDetId(1, 1, 0, 1, 1, 0))->nEtaPartitions() != GEMPadDigi::NumberPartitions::ME0) {
      edm::LogError("GEMPadDigiProducer") << "ME0 geometry appears corrupted";
    }
  }

  // check that GE1/1 has 8-eta partitions
  if (geometry_->hasGE11()) {
    if (geometry_->chamber(GEMDetId(1, 1, 1, 1, 1, 0))->nEtaPartitions() != GEMPadDigi::NumberPartitions::GE11) {
      edm::LogError("GEMPadDigiProducer") << "GE1/1 geometry appears corrupted";
    }
  }

  if (geometry_->hasGE21()) {
    if (!use16GE21_) {
      // check that GE2/1 has 8-eta partitions
      if (geometry_->chamber(GEMDetId(1, 1, 2, 1, 1, 0))->nEtaPartitions() != GEMPadDigi::NumberPartitions::GE21) {
        edm::LogError("GEMPadDigiProducer") << "GE2/1 geometry (8 partition) appears corrupted";
      }
    } else {
      // check that GE2/1 has 16-eta partitions
      if (geometry_->chamber(GEMDetId(1, 1, 2, 1, 1, 0))->nEtaPartitions() !=
          GEMPadDigi::NumberPartitions::GE21SplitStrip) {
        edm::LogError("GEMPadDigiProducer") << "GE2/1 geometry (16 partition) appears corrupted";
      }
    }
  }
}

DEFINE_FWK_MODULE(GEMPadDigiProducer);
