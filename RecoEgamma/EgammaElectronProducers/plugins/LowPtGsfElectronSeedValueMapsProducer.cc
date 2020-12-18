#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/transform.h"

#include <vector>
#include <string>

class LowPtGsfElectronSeedValueMapsProducer : public edm::stream::EDProducer<> {
public:
  explicit LowPtGsfElectronSeedValueMapsProducer(const edm::ParameterSet&);

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  edm::EDGetTokenT<reco::GsfTrackCollection> gsfTracks_;
  edm::EDGetTokenT<edm::ValueMap<reco::PreIdRef> > preIdsValueMap_;
  std::vector<std::string> names_;
  const bool rekey_;
  edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectrons_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > floatValueMaps_;
};

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSeedValueMapsProducer::LowPtGsfElectronSeedValueMapsProducer(const edm::ParameterSet& conf)
    : gsfTracks_(),
      preIdsValueMap_(),
      names_(),
      rekey_(conf.getParameter<bool>("rekey")),
      gsfElectrons_(),
      floatValueMaps_() {
  if (rekey_) {
    gsfElectrons_ = consumes<reco::GsfElectronCollection>(conf.getParameter<edm::InputTag>("gsfElectrons"));
    std::vector<edm::InputTag> tags = conf.getParameter<std::vector<edm::InputTag> >("floatValueMaps");
    for (const auto& tag : tags) {
      floatValueMaps_ = edm::vector_transform(
          tags, [this](edm::InputTag const& tag) { return consumes<edm::ValueMap<float> >(tag); });
      names_.push_back(tag.instance());
      produces<edm::ValueMap<float> >(tag.instance());
    }
  } else {
    gsfTracks_ = consumes<reco::GsfTrackCollection>(conf.getParameter<edm::InputTag>("gsfTracks"));
    preIdsValueMap_ = consumes<edm::ValueMap<reco::PreIdRef> >(conf.getParameter<edm::InputTag>("preIdsValueMap"));
    names_ = conf.getParameter<std::vector<std::string> >("ModelNames");
    for (const auto& name : names_) {
      produces<edm::ValueMap<float> >(name);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedValueMapsProducer::produce(edm::Event& event, const edm::EventSetup&) {
  if (rekey_ == false) {
    // TRANSFORM VALUEMAP OF PREID OBJECTS KEYED BY KF TRACK ...
    // .. INTO VALUEMAP OF FLOATS (BDT SCORE) KEYED BY GSF TRACK ...

    // Retrieve GsfTracks from Event
    auto gsfTracks = event.getHandle(gsfTracks_);

    // Retrieve PreIds from Event
    auto preIdsValueMap = event.getHandle(preIdsValueMap_);

    // Iterate through GsfTracks, extract BDT output, and store result in ValueMap for each model
    std::vector<std::vector<float> > output;
    for (unsigned int iname = 0; iname < names_.size(); ++iname) {
      output.push_back(std::vector<float>(gsfTracks->size(), -999.));
    }
    auto const& gsfTracksV = *gsfTracks;
    for (unsigned int igsf = 0; igsf < gsfTracksV.size(); igsf++) {
      const reco::GsfTrack& gsf = gsfTracksV[igsf];
      if (gsf.extra().isNonnull() && gsf.extra()->seedRef().isNonnull()) {
        reco::ElectronSeedRef seed = gsf.extra()->seedRef().castTo<reco::ElectronSeedRef>();
        if (seed.isNonnull() && seed->ctfTrack().isNonnull()) {
          const reco::PreIdRef preid = (*preIdsValueMap)[seed->ctfTrack()];
          if (preid.isNonnull()) {
            for (unsigned int iname = 0; iname < names_.size(); ++iname) {
              output[iname][igsf] = preid->mva(iname);
            }
          }
        }
      }
    }

    // Create and put ValueMap in Event
    for (unsigned int iname = 0; iname < names_.size(); ++iname) {
      auto ptr = std::make_unique<edm::ValueMap<float> >(edm::ValueMap<float>());
      edm::ValueMap<float>::Filler filler(*ptr);
      filler.insert(gsfTracks, output[iname].begin(), output[iname].end());
      filler.fill();
      event.put(std::move(ptr), names_[iname]);
    }

  } else {
    // TRANSFORM VALUEMAP OF FLOATS (BDT SCORE) KEYED BY GSF TRACK ...
    // .. INTO VALUEMAP OF FLOATS (BDT SCORE) KEYED BY GSF ELECTRON ...

    // Retrieve GsfElectrons from Event
    auto gsfElectrons = event.getHandle(gsfElectrons_);

    // Retrieve float ValueMaps from Event
    for (unsigned int idx = 0; idx < names_.size(); ++idx) {
      // Extract ValueMap from Event
      auto const& floatValueMap = event.get(floatValueMaps_[idx]);

      // Store BDT scores in vector
      std::vector<float> output(gsfElectrons->size(), -99.);
      auto const& gsfElectronsV = *gsfElectrons;
      for (unsigned int iele = 0; iele < gsfElectronsV.size(); iele++) {
        const reco::GsfElectron& ele = gsfElectronsV[iele];
        reco::GsfTrackRef gsf = ele.gsfTrack();
        output[iele] = floatValueMap[gsf];
      }
      // Create and put ValueMap in Event
      auto ptr = std::make_unique<edm::ValueMap<float> >(edm::ValueMap<float>());
      edm::ValueMap<float>::Filler filler(*ptr);
      filler.insert(gsfElectrons, output.begin(), output.end());
      filler.fill();
      event.put(std::move(ptr), names_[idx]);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedValueMapsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfTracks", edm::InputTag("lowPtGsfEleGsfTracks"));
  desc.add<edm::InputTag>("preIdsValueMap", edm::InputTag("lowPtGsfElectronSeeds"));
  desc.add<std::vector<std::string> >("ModelNames", {"unbiased", "ptbiased"});
  desc.add<bool>("rekey", false);
  desc.add<edm::InputTag>("gsfElectrons", edm::InputTag());
  desc.add<std::vector<edm::InputTag> >("floatValueMaps", std::vector<edm::InputTag>());
  descriptions.add("lowPtGsfElectronSeedValueMaps", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronSeedValueMapsProducer);
