#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
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

class ME0PadDigiProducer : public edm::global::EDProducer<> {
public:
  explicit ME0PadDigiProducer(const edm::ParameterSet& ps);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  ME0PadDigiCollection buildPads(const ME0DigiCollection& digis, const ME0Geometry& geometry) const;

  /// Name of input digi Collection
  edm::EDGetTokenT<ME0DigiCollection> digi_token_;
  edm::InputTag digis_;
  edm::ESGetToken<ME0Geometry, MuonGeometryRecord> geom_token_;
  edm::EDPutTokenT<ME0PadDigiCollection> put_token_;
};

ME0PadDigiProducer::ME0PadDigiProducer(const edm::ParameterSet& ps) {
  digis_ = ps.getParameter<edm::InputTag>("InputCollection");

  digi_token_ = consumes<ME0DigiCollection>(digis_);
  geom_token_ = esConsumes<ME0Geometry, MuonGeometryRecord>();

  put_token_ = produces<ME0PadDigiCollection>();
}

void ME0PadDigiProducer::produce(edm::StreamID, edm::Event& e, const edm::EventSetup& eventSetup) const {
  auto const& geometry = eventSetup.getData(geom_token_);

  edm::Handle<ME0DigiCollection> hdigis;
  e.getByToken(digi_token_, hdigis);

  // build the pads and store them in the event
  e.emplace(put_token_, buildPads(*(hdigis.product()), geometry));
}

ME0PadDigiCollection ME0PadDigiProducer::buildPads(const ME0DigiCollection& det_digis,
                                                   const ME0Geometry& geometry) const {
  ME0PadDigiCollection out_pads;
  for (const auto& p : geometry.etaPartitions()) {
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
  return out_pads;
}

DEFINE_FWK_MODULE(ME0PadDigiProducer);
