#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include <set>

/// \class ME0PadDigiProducer

class ME0PadDigiProducer : public edm::stream::EDProducer<> {
public:
  explicit ME0PadDigiProducer(const edm::ParameterSet& ps);

  ~ME0PadDigiProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  void buildPads(const ME0DigiCollection& digis, ME0PadDigiCollection& out_pads) const;

  /// Name of input digi Collection
  edm::EDGetTokenT<ME0DigiCollection> digi_token_;
  edm::InputTag digis_;

  const ME0Geometry* geometry_;
};

ME0PadDigiProducer::ME0PadDigiProducer(const edm::ParameterSet& ps) : geometry_(nullptr) {
  digis_ = ps.getParameter<edm::InputTag>("InputCollection");

  digi_token_ = consumes<ME0DigiCollection>(digis_);

  produces<ME0PadDigiCollection>();
}

ME0PadDigiProducer::~ME0PadDigiProducer() {}

void ME0PadDigiProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  edm::ESHandle<ME0Geometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_ = &*hGeom;
}

void ME0PadDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Handle<ME0DigiCollection> hdigis;
  e.getByToken(digi_token_, hdigis);

  // Create empty output
  std::unique_ptr<ME0PadDigiCollection> pPads(new ME0PadDigiCollection());

  // build the pads
  buildPads(*(hdigis.product()), *pPads);

  // store them in the event
  e.put(std::move(pPads));
}

void ME0PadDigiProducer::buildPads(const ME0DigiCollection& det_digis, ME0PadDigiCollection& out_pads) const {
  for (const auto& p : geometry_->etaPartitions()) {
    // set of <pad, bx> pairs, sorted first by pad then by bx
    std::set<std::pair<int, int> > proto_pads;

    // walk over digis in this partition,
    // and stuff them into a set of unique pads (equivalent of OR operation)
    auto digis = det_digis.get(p->id());
    for (auto d = digis.first; d != digis.second; ++d) {
      int pad_num = 1 + static_cast<int>(p->padOfStrip(d->strip()));
      proto_pads.emplace(pad_num, d->bx());
    }

    // fill the output collections
    for (const auto& d : proto_pads) {
      ME0PadDigi pad_digi(d.first, d.second);
      out_pads.insertDigi(p->id(), pad_digi);
    }
  }
}

DEFINE_FWK_MODULE(ME0PadDigiProducer);
