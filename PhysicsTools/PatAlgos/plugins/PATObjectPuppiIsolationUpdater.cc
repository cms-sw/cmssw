#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace pat {
  template <typename T>
  class PATObjectPuppiIsolationUpdater : public edm::stream::EDProducer<> {
  public:
    explicit PATObjectPuppiIsolationUpdater(const edm::ParameterSet &iConfig);
    ~PATObjectPuppiIsolationUpdater() override;

    void produce(edm::Event &, const edm::EventSetup &) override;

  private:
    // configurables
    edm::EDGetTokenT<std::vector<T>> src_;
    //PUPPI isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPIIsolation_photons_;
    //PUPPINoLeptons isolation tokens
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_charged_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_neutral_hadrons_;
    edm::EDGetTokenT<edm::ValueMap<float>> PUPPINoLeptonsIsolation_photons_;
  };
}  // namespace pat

using namespace pat;

template <typename T>
PATObjectPuppiIsolationUpdater<T>::PATObjectPuppiIsolationUpdater(const edm::ParameterSet &iConfig)
    : src_(consumes<std::vector<T>>(iConfig.getParameter<edm::InputTag>("src"))) {
  PUPPIIsolation_charged_hadrons_ =
      consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationChargedHadrons"));
  PUPPIIsolation_neutral_hadrons_ =
      consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationNeutralHadrons"));
  PUPPIIsolation_photons_ =
      consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiIsolationPhotons"));
  if constexpr (std::is_same<T, pat::Electron>::value || std::is_same<T, pat::Muon>::value) {
    PUPPINoLeptonsIsolation_charged_hadrons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiNoLeptonsIsolationChargedHadrons"));
    PUPPINoLeptonsIsolation_neutral_hadrons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiNoLeptonsIsolationNeutralHadrons"));
    PUPPINoLeptonsIsolation_photons_ =
        consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiNoLeptonsIsolationPhotons"));
  }
  produces<std::vector<T>>();
}

template <typename T>
PATObjectPuppiIsolationUpdater<T>::~PATObjectPuppiIsolationUpdater() {}

template <typename T>
void PATObjectPuppiIsolationUpdater<T>::produce(edm::Event &iEvent, edm::EventSetup const &) {
  edm::Handle<std::vector<T>> src;
  iEvent.getByToken(src_, src);

  //value maps for puppi isolation
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_charged_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_neutral_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPIIsolation_photons;
  //value maps for puppiNoLeptons isolation
  edm::Handle<edm::ValueMap<float>> PUPPINoLeptonsIsolation_charged_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPINoLeptonsIsolation_neutral_hadrons;
  edm::Handle<edm::ValueMap<float>> PUPPINoLeptonsIsolation_photons;

  //puppi
  iEvent.getByToken(PUPPIIsolation_charged_hadrons_, PUPPIIsolation_charged_hadrons);
  iEvent.getByToken(PUPPIIsolation_neutral_hadrons_, PUPPIIsolation_neutral_hadrons);
  iEvent.getByToken(PUPPIIsolation_photons_, PUPPIIsolation_photons);
  //puppiNoLeptons

  if constexpr (std::is_same<T, pat::Electron>::value || std::is_same<T, pat::Muon>::value) {
    iEvent.getByToken(PUPPINoLeptonsIsolation_charged_hadrons_, PUPPINoLeptonsIsolation_charged_hadrons);
    iEvent.getByToken(PUPPINoLeptonsIsolation_neutral_hadrons_, PUPPINoLeptonsIsolation_neutral_hadrons);
    iEvent.getByToken(PUPPINoLeptonsIsolation_photons_, PUPPINoLeptonsIsolation_photons);
  }

  auto outPtrP = std::make_unique<std::vector<T>>();
  outPtrP->reserve(src->size());

  for (size_t i = 0; i < src->size(); ++i) {
    // copy original pat object and append to vector
    outPtrP->emplace_back((*src)[i]);

    edm::Ptr<T> objPtr(src, i);

    outPtrP->back().setIsolationPUPPI((*PUPPIIsolation_charged_hadrons)[objPtr],
                                      (*PUPPIIsolation_neutral_hadrons)[objPtr],
                                      (*PUPPIIsolation_photons)[objPtr]);

    if constexpr (std::is_same<T, pat::Electron>::value || std::is_same<T, pat::Muon>::value) {
      outPtrP->back().setIsolationPUPPINoLeptons((*PUPPINoLeptonsIsolation_charged_hadrons)[objPtr],
                                                 (*PUPPINoLeptonsIsolation_neutral_hadrons)[objPtr],
                                                 (*PUPPINoLeptonsIsolation_photons)[objPtr]);
    }
  }
  iEvent.put(std::move(outPtrP));
}

typedef PATObjectPuppiIsolationUpdater<pat::Electron> PATElectronPuppiIsolationUpdater;
typedef PATObjectPuppiIsolationUpdater<pat::Photon> PATPhotonPuppiIsolationUpdater;
typedef PATObjectPuppiIsolationUpdater<pat::Muon> PATMuonPuppiIsolationUpdater;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATElectronPuppiIsolationUpdater);
DEFINE_FWK_MODULE(PATPhotonPuppiIsolationUpdater);
DEFINE_FWK_MODULE(PATMuonPuppiIsolationUpdater);
