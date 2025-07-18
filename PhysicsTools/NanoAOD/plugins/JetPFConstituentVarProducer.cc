#include <string>
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

template <typename T>
class JetPFConstituentVarProducer : public edm::stream::EDProducer<> {
public:
  explicit JetPFConstituentVarProducer(const edm::ParameterSet&);
  ~JetPFConstituentVarProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void PutValueMapInEvent(edm::Event&, const edm::Handle<edm::View<T>>&, const std::vector<float>&, std::string);

  edm::EDGetTokenT<edm::View<T>> jet_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> puppi_value_map_token_;

  const bool fallback_puppi_weight_;
  bool use_puppi_value_map_;
};

//
// constructors and destructor
//
template <typename T>
JetPFConstituentVarProducer<T>::JetPFConstituentVarProducer(const edm::ParameterSet& iConfig)
    : jet_token_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("jets"))),
      fallback_puppi_weight_(iConfig.getParameter<bool>("fallback_puppi_weight")),
      use_puppi_value_map_(false) {
  //
  // Puppi Value Map
  //
  const auto& puppi_value_map_tag = iConfig.getParameter<edm::InputTag>("puppi_value_map");
  if (!puppi_value_map_tag.label().empty()) {
    puppi_value_map_token_ = consumes<edm::ValueMap<float>>(puppi_value_map_tag);
    use_puppi_value_map_ = true;
  }
  produces<edm::ValueMap<float>>("leadConstNeHadEF");
  produces<edm::ValueMap<float>>("leadConstChHadEF");
  produces<edm::ValueMap<float>>("leadConstPhotonEF");
  produces<edm::ValueMap<float>>("leadConstElectronEF");
  produces<edm::ValueMap<float>>("leadConstMuonEF");
  produces<edm::ValueMap<float>>("leadConstHFHADEF");
  produces<edm::ValueMap<float>>("leadConstHFEMEF");
  produces<edm::ValueMap<float>>("leadConstNeHadPuppiWeight");
  produces<edm::ValueMap<float>>("leadConstChHadPuppiWeight");
  produces<edm::ValueMap<float>>("leadConstPhotonPuppiWeight");
  produces<edm::ValueMap<float>>("leadConstElectronPuppiWeight");
  produces<edm::ValueMap<float>>("leadConstMuonPuppiWeight");
  produces<edm::ValueMap<float>>("leadConstHFHADPuppiWeight");
  produces<edm::ValueMap<float>>("leadConstHFEMPuppiWeight");
}

template <typename T>
JetPFConstituentVarProducer<T>::~JetPFConstituentVarProducer() {}

template <typename T>
void JetPFConstituentVarProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Input jets
  auto jets = iEvent.getHandle(jet_token_);

  // Get puppi value map
  edm::Handle<edm::ValueMap<float>> puppi_value_map_;
  if (use_puppi_value_map_) {
    iEvent.getByToken(puppi_value_map_token_, puppi_value_map_);
  }

  std::vector<float> jet_leadConstNeHadEF(jets->size(), 0.f);
  std::vector<float> jet_leadConstNeHadPuppiWeight(jets->size(), 0.f);

  std::vector<float> jet_leadConstChHadEF(jets->size(), 0.f);
  std::vector<float> jet_leadConstChHadPuppiWeight(jets->size(), 0.f);

  std::vector<float> jet_leadConstPhotonEF(jets->size(), 0.f);
  std::vector<float> jet_leadConstPhotonPuppiWeight(jets->size(), 0.f);

  std::vector<float> jet_leadConstElectronEF(jets->size(), 0.f);
  std::vector<float> jet_leadConstElectronPuppiWeight(jets->size(), 0.f);

  std::vector<float> jet_leadConstMuonEF(jets->size(), 0.f);
  std::vector<float> jet_leadConstMuonPuppiWeight(jets->size(), 0.f);

  std::vector<float> jet_leadConstHFHADEF(jets->size(), 0.f);
  std::vector<float> jet_leadConstHFHADPuppiWeight(jets->size(), 0.f);

  std::vector<float> jet_leadConstHFEMEF(jets->size(), 0.f);
  std::vector<float> jet_leadConstHFEMPuppiWeight(jets->size(), 0.f);

  // Loop over jet
  for (std::size_t jet_idx = 0; jet_idx < jets->size(); jet_idx++) {
    const auto& jet = (*jets)[jet_idx];

    float jet_energy_raw = jet.energy();
    if constexpr (std::is_same<T, pat::Jet>::value) {
      jet_energy_raw = jet.correctedJet(0).energy();
    }

    reco::CandidatePtr leadConstNeHad;
    reco::CandidatePtr leadConstChHad;
    reco::CandidatePtr leadConstPhoton;
    reco::CandidatePtr leadConstElectron;
    reco::CandidatePtr leadConstMuon;
    reco::CandidatePtr leadConstHFHAD;
    reco::CandidatePtr leadConstHFEM;

    float leadConstNeHadPuppiWeight = 1.f;
    float leadConstChHadPuppiWeight = 1.f;
    float leadConstPhotonPuppiWeight = 1.f;
    float leadConstElectronPuppiWeight = 1.f;
    float leadConstMuonPuppiWeight = 1.f;
    float leadConstHFHADPuppiWeight = 1.f;
    float leadConstHFEMPuppiWeight = 1.f;

    //
    // Loop over jet constituents
    //
    for (const reco::CandidatePtr& dau : jet.daughterPtrVector()) {
      float puppiw = 1.f;

      //
      // Get Puppi weight from ValueMap, if provided.
      //
      if (use_puppi_value_map_) {
        puppiw = (*puppi_value_map_)[dau];
      } else if (!fallback_puppi_weight_) {
        throw edm::Exception(edm::errors::InvalidReference, "PUPPI value map missing")
            << "use fallback_puppi_weight option to use " << puppiw << " for cand as default";
      }

      //
      // Find the highest energy constituents for each PF type
      //
      if (abs(dau->pdgId()) == 130) {
        if (leadConstNeHad.isNull() ||
            (puppiw * dau->energy() > leadConstNeHadPuppiWeight * leadConstNeHad->energy())) {
          leadConstNeHad = dau;
          leadConstNeHadPuppiWeight = puppiw;
        }
      } else if (abs(dau->pdgId()) == 211) {
        if (leadConstChHad.isNull() ||
            (puppiw * dau->energy() > leadConstChHadPuppiWeight * leadConstChHad->energy())) {
          leadConstChHad = dau;
          leadConstChHadPuppiWeight = puppiw;
        }
      } else if (abs(dau->pdgId()) == 22) {
        if (leadConstPhoton.isNull() ||
            (puppiw * dau->energy() > leadConstPhotonPuppiWeight * leadConstPhoton->energy())) {
          leadConstPhoton = dau;
          leadConstPhotonPuppiWeight = puppiw;
        }
      } else if (abs(dau->pdgId()) == 11) {
        if (leadConstElectron.isNull() ||
            (puppiw * dau->energy() > leadConstElectronPuppiWeight * leadConstElectron->energy())) {
          leadConstElectron = dau;
          leadConstElectronPuppiWeight = puppiw;
        }
      } else if (abs(dau->pdgId()) == 13) {
        if (leadConstMuon.isNull() || (puppiw * dau->energy() > leadConstMuonPuppiWeight * leadConstMuon->energy())) {
          leadConstMuon = dau;
          leadConstMuonPuppiWeight = puppiw;
        }
      } else if (abs(dau->pdgId()) == 1) {
        if (leadConstHFHAD.isNull() ||
            (puppiw * dau->energy() > leadConstHFHADPuppiWeight * leadConstHFHAD->energy())) {
          leadConstHFHAD = dau;
          leadConstHFHADPuppiWeight = puppiw;
        }
      } else if (abs(dau->pdgId()) == 2) {
        if (leadConstHFEM.isNull() || (puppiw * dau->energy() > leadConstHFEMPuppiWeight * leadConstHFEM->energy())) {
          leadConstHFEM = dau;
          leadConstHFEMPuppiWeight = puppiw;
        }
      }
    }  // End of Jet Constituents Loop

    if (leadConstNeHad.isNonnull()) {
      jet_leadConstNeHadEF[jet_idx] = (leadConstNeHad->energy() * leadConstNeHadPuppiWeight) / jet_energy_raw;
      jet_leadConstNeHadPuppiWeight[jet_idx] = leadConstNeHadPuppiWeight;
    }
    if (leadConstChHad.isNonnull()) {
      jet_leadConstChHadEF[jet_idx] = (leadConstChHad->energy() * leadConstChHadPuppiWeight) / jet_energy_raw;
      jet_leadConstChHadPuppiWeight[jet_idx] = leadConstChHadPuppiWeight;
    }
    if (leadConstPhoton.isNonnull()) {
      jet_leadConstPhotonEF[jet_idx] = (leadConstPhoton->energy() * leadConstPhotonPuppiWeight) / jet_energy_raw;
      jet_leadConstPhotonPuppiWeight[jet_idx] = leadConstPhotonPuppiWeight;
    }
    if (leadConstElectron.isNonnull()) {
      jet_leadConstElectronEF[jet_idx] = (leadConstElectron->energy() * leadConstElectronPuppiWeight) / jet_energy_raw;
      jet_leadConstElectronPuppiWeight[jet_idx] = leadConstElectronPuppiWeight;
    }
    if (leadConstMuon.isNonnull()) {
      jet_leadConstMuonEF[jet_idx] = (leadConstMuon->energy() * leadConstMuonPuppiWeight) / jet_energy_raw;
      jet_leadConstMuonPuppiWeight[jet_idx] = leadConstMuonPuppiWeight;
    }
    if (leadConstHFHAD.isNonnull()) {
      jet_leadConstHFHADEF[jet_idx] = (leadConstHFHAD->energy() * leadConstHFHADPuppiWeight) / jet_energy_raw;
      jet_leadConstHFHADPuppiWeight[jet_idx] = leadConstHFHADPuppiWeight;
    }
    if (leadConstHFEM.isNonnull()) {
      jet_leadConstHFEMEF[jet_idx] = (leadConstHFEM->energy() * leadConstHFEMPuppiWeight) / jet_energy_raw;
      jet_leadConstHFEMPuppiWeight[jet_idx] = leadConstHFEMPuppiWeight;
    }
  }

  PutValueMapInEvent(iEvent, jets, jet_leadConstNeHadEF, "leadConstNeHadEF");
  PutValueMapInEvent(iEvent, jets, jet_leadConstNeHadPuppiWeight, "leadConstNeHadPuppiWeight");

  PutValueMapInEvent(iEvent, jets, jet_leadConstChHadEF, "leadConstChHadEF");
  PutValueMapInEvent(iEvent, jets, jet_leadConstChHadPuppiWeight, "leadConstChHadPuppiWeight");

  PutValueMapInEvent(iEvent, jets, jet_leadConstPhotonEF, "leadConstPhotonEF");
  PutValueMapInEvent(iEvent, jets, jet_leadConstPhotonPuppiWeight, "leadConstPhotonPuppiWeight");

  PutValueMapInEvent(iEvent, jets, jet_leadConstElectronEF, "leadConstElectronEF");
  PutValueMapInEvent(iEvent, jets, jet_leadConstElectronPuppiWeight, "leadConstElectronPuppiWeight");

  PutValueMapInEvent(iEvent, jets, jet_leadConstMuonEF, "leadConstMuonEF");
  PutValueMapInEvent(iEvent, jets, jet_leadConstMuonPuppiWeight, "leadConstMuonPuppiWeight");

  PutValueMapInEvent(iEvent, jets, jet_leadConstHFHADEF, "leadConstHFHADEF");
  PutValueMapInEvent(iEvent, jets, jet_leadConstHFHADPuppiWeight, "leadConstHFHADPuppiWeight");

  PutValueMapInEvent(iEvent, jets, jet_leadConstHFEMEF, "leadConstHFEMEF");
  PutValueMapInEvent(iEvent, jets, jet_leadConstHFEMPuppiWeight, "leadConstHFEMPuppiWeight");
}
template <typename T>
void JetPFConstituentVarProducer<T>::PutValueMapInEvent(edm::Event& iEvent,
                                                        const edm::Handle<edm::View<T>>& coll,
                                                        const std::vector<float>& vec_var,
                                                        std::string VMName) {
  std::unique_ptr<edm::ValueMap<float>> VM(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerVM(*VM);
  fillerVM.insert(coll, vec_var.begin(), vec_var.end());
  fillerVM.fill();
  iEvent.put(std::move(VM), VMName);
}

template <typename T>
void JetPFConstituentVarProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("finalJetsPuppi"));
  desc.add<edm::InputTag>("puppi_value_map", edm::InputTag("packedpuppi"));
  desc.add<bool>("fallback_puppi_weight", false);
  descriptions.addWithDefaultLabel(desc);
}

typedef JetPFConstituentVarProducer<pat::Jet> PatJetPFConstituentVarProducer;
DEFINE_FWK_MODULE(PatJetPFConstituentVarProducer);
