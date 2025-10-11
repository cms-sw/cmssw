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
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

class PFCandidatesVarProducer : public edm::stream::EDProducer<> {
public:
  explicit PFCandidatesVarProducer(const edm::ParameterSet&);
  ~PFCandidatesVarProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void PutValueMapInEvent(edm::Event&,
                          const edm::Handle<edm::View<reco::Candidate>>&,
                          const std::vector<float>&,
                          std::string);
  edm::EDGetTokenT<edm::View<reco::Candidate>> pfcands_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> puppi_value_map_token_;
};

PFCandidatesVarProducer::PFCandidatesVarProducer(const edm::ParameterSet& iConfig)
    : pfcands_token_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("src"))),
      puppi_value_map_token_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppi_value_map"))) {
  produces<edm::ValueMap<float>>("ptWeighted");
  produces<edm::ValueMap<float>>("massWeighted");
}

PFCandidatesVarProducer::~PFCandidatesVarProducer() {}

void PFCandidatesVarProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get PF candidates
  edm::Handle<edm::View<reco::Candidate>> pfcands_handle;
  iEvent.getByToken(pfcands_token_, pfcands_handle);
  const edm::View<reco::Candidate>* pfcands = pfcands_handle.product();

  // Get puppi value map
  edm::Handle<edm::ValueMap<float>> puppi_value_map_;
  iEvent.getByToken(puppi_value_map_token_, puppi_value_map_);

  std::vector<float> ptWeighted(pfcands->size(), 0.f);
  std::vector<float> massWeighted(pfcands->size(), 0.f);

  for (std::size_t pfcand_idx = 0; pfcand_idx < pfcands->size(); pfcand_idx++) {
    const auto& cand = (*pfcands)[pfcand_idx];

    reco::CandidatePtr candPtr(pfcands_handle, pfcand_idx);
    float puppiWeightVal = (*puppi_value_map_)[candPtr];

    ptWeighted[pfcand_idx] = puppiWeightVal * cand.pt();
    massWeighted[pfcand_idx] = puppiWeightVal * cand.mass();
  }

  PutValueMapInEvent(iEvent, pfcands_handle, ptWeighted, "ptWeighted");
  PutValueMapInEvent(iEvent, pfcands_handle, massWeighted, "massWeighted");
}
void PFCandidatesVarProducer::PutValueMapInEvent(edm::Event& iEvent,
                                                 const edm::Handle<edm::View<reco::Candidate>>& coll,
                                                 const std::vector<float>& vec_var,
                                                 std::string VMName) {
  std::unique_ptr<edm::ValueMap<float>> VM(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerVM(*VM);
  fillerVM.insert(coll, vec_var.begin(), vec_var.end());
  fillerVM.fill();
  iEvent.put(std::move(VM), VMName);
}

void PFCandidatesVarProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("puppi_value_map", edm::InputTag("packedpuppi"));
  desc.add<bool>("fallback_puppi_weight", false);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PFCandidatesVarProducer);
