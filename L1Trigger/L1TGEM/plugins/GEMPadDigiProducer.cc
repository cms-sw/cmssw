#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
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

  /// Name of input digi Collection
  edm::EDGetTokenT<GEMDigiCollection> digi_token_;
  edm::InputTag digis_;

  const GEMGeometry* geometry_;
};

GEMPadDigiProducer::GEMPadDigiProducer(const edm::ParameterSet& ps) : geometry_(nullptr) {
  digis_ = ps.getParameter<edm::InputTag>("InputCollection");

  digi_token_ = consumes<GEMDigiCollection>(digis_);

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
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_ = &*hGeom;
}

void GEMPadDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Handle<GEMDigiCollection> hdigis;
  e.getByToken(digi_token_, hdigis);

  // Create empty output
  std::unique_ptr<GEMPadDigiCollection> pPads(new GEMPadDigiCollection());

  // build the pads
  buildPads(*(hdigis.product()), *pPads);

  // store them in the event
  e.put(std::move(pPads));
}

void GEMPadDigiProducer::buildPads(const GEMDigiCollection& det_digis, GEMPadDigiCollection& out_pads) const {
  for (const auto& p : geometry_->etaPartitions()) {
    // set of <pad, bx> pairs, sorted first by pad then by bx
    std::set<std::pair<int, int> > proto_pads;

    // walk over digis in this partition,
    // and stuff them into a set of unique pads (equivalent of OR operation)
    auto digis = det_digis.get(p->id());
    for (auto d = digis.first; d != digis.second; ++d) {
      int pad_num = static_cast<int>(p->padOfStrip(d->strip()));
      proto_pads.emplace(pad_num, d->bx());
    }

    // fill the output collections
    for (const auto& d : proto_pads) {
      GEMPadDigi pad_digi(d.first, d.second);
      out_pads.insertDigi(p->id(), pad_digi);
    }
  }
}

DEFINE_FWK_MODULE(GEMPadDigiProducer);
