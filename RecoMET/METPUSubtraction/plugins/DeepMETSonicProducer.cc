#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "RecoMET/METPUSubtraction/interface/DeepMETHelp.h"

using namespace deepmet_helper;

class DeepMETSonicProducer : public TritonEDProducer<> {
public:
  explicit DeepMETSonicProducer(const edm::ParameterSet&);
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<std::vector<pat::PackedCandidate>> pf_token_;
  const float norm_;
  const bool ignore_leptons_;
  const unsigned int max_n_pf_;
  const float scale_;
  float px_leptons_;
  float py_leptons_;
};

DeepMETSonicProducer::DeepMETSonicProducer(const edm::ParameterSet& cfg)
    : TritonEDProducer<>(cfg),
      pf_token_(consumes<std::vector<pat::PackedCandidate>>(cfg.getParameter<edm::InputTag>("pf_src"))),
      norm_(cfg.getParameter<double>("norm_factor")),
      ignore_leptons_(cfg.getParameter<bool>("ignore_leptons")),
      max_n_pf_(cfg.getParameter<unsigned int>("max_n_pf")),
      scale_(1.0 / norm_) {
  produces<pat::METCollection>();
}

void DeepMETSonicProducer::acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) {
  // one event per batch
  client_->setBatchSize(1);
  px_leptons_ = 0.;
  py_leptons_ = 0.;

  auto const& pfs = iEvent.get(pf_token_);

  auto& input = iInput.at("input");
  auto pfdata = input.allocate<float>();
  auto& vpfdata = (*pfdata)[0];

  auto& input_cat0 = iInput.at("input_cat0");
  auto pfchg = input_cat0.allocate<float>();
  auto& vpfchg = (*pfchg)[0];

  auto& input_cat1 = iInput.at("input_cat1");
  auto pfpdgId = input_cat1.allocate<float>();
  auto& vpfpdgId = (*pfpdgId)[0];

  auto& input_cat2 = iInput.at("input_cat2");
  auto pffromPV = input_cat2.allocate<float>();
  auto& vpffromPV = (*pffromPV)[0];

  size_t i_pf = 0;
  for (const auto& pf : pfs) {
    if (ignore_leptons_) {
      int pdg_id = std::abs(pf.pdgId());
      if (pdg_id == 11 || pdg_id == 13) {
        px_leptons_ += pf.px();
        py_leptons_ += pf.py();
        continue;
      }
    }

    // PF keys [b'PF_dxy', b'PF_dz', b'PF_eta', b'PF_mass', b'PF_pt', b'PF_puppiWeight', b'PF_px', b'PF_py']
    vpfdata.push_back(pf.dxy());
    vpfdata.push_back(pf.dz());
    vpfdata.push_back(pf.eta());
    vpfdata.push_back(pf.mass());
    vpfdata.push_back(scale_and_rm_outlier(pf.pt(), scale_));
    vpfdata.push_back(pf.puppiWeight());
    vpfdata.push_back(scale_and_rm_outlier(pf.px(), scale_));
    vpfdata.push_back(scale_and_rm_outlier(pf.py(), scale_));

    vpfchg.push_back(charge_embedding.at(pf.charge()));

    vpfpdgId.push_back(pdg_id_embedding.at(pf.pdgId()));

    vpffromPV.push_back(pf.fromPV());

    ++i_pf;
    if (i_pf == max_n_pf_) {
      edm::LogWarning("acquire")
          << "<DeepMETSonicProducer::acquire>:" << std::endl
          << " The number of particles is equal to or exceeds the maximum considerable for DeepMET" << std::endl;
      break;
    }
  }

  // fill the remaining with zeros
  // resize the vector to 4500 for zero-padding
  vpfdata.resize(8 * max_n_pf_);
  vpfchg.resize(max_n_pf_);
  vpfpdgId.resize(max_n_pf_);
  vpffromPV.resize(max_n_pf_);

  input.toServer(pfdata);
  input_cat0.toServer(pfchg);
  input_cat1.toServer(pfpdgId);
  input_cat2.toServer(pffromPV);
}

void DeepMETSonicProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) {
  const auto& output1 = iOutput.begin()->second;
  const auto& outputs = output1.fromServer<float>();

  // outputs are px and py
  float px = outputs[0][0] * norm_;
  float py = outputs[0][1] * norm_;

  // subtract the lepton pt contribution
  px -= px_leptons_;
  py -= py_leptons_;

  LogDebug("produce") << "<DeepMETSonicProducer::produce>:" << std::endl
                      << " MET from DeepMET Sonic Producer is MET_x " << px << " and MET_y " << py << std::endl;

  auto pf_mets = std::make_unique<pat::METCollection>();
  const reco::Candidate::LorentzVector p4(px, py, 0., std::hypot(px, py));
  pf_mets->emplace_back(reco::MET(p4, {}));
  iEvent.put(std::move(pf_mets));
}

void DeepMETSonicProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  TritonClient::fillPSetDescription(desc);
  desc.add<edm::InputTag>("pf_src", edm::InputTag("packedPFCandidates"));
  desc.add<bool>("ignore_leptons", false);
  desc.add<double>("norm_factor", 50.);
  desc.add<unsigned int>("max_n_pf", 4500);
  descriptions.add("deepMETSonicProducer", desc);
}

DEFINE_FWK_MODULE(DeepMETSonicProducer);
