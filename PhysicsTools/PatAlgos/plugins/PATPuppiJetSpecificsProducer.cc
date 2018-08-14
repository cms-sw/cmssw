/*
 * PATPuppiJetSpecificProducer
 *
 * Author: Andreas Hinzmann
 *
 * Compute weighted constituent multiplicites for PUPPI PAT jets
 *
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"

class PATPuppiJetSpecificProducer : public edm::global::EDProducer<> 
{
public:

  explicit PATPuppiJetSpecificProducer(const edm::ParameterSet& cfg);
  ~PATPuppiJetSpecificProducer() {}
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  // input collection
  edm::InputTag srcjets_;
  edm::EDGetTokenT<edm::View<pat::Jet> > jets_token;

};

PATPuppiJetSpecificProducer::PATPuppiJetSpecificProducer(const edm::ParameterSet& cfg)
{
  srcjets_ = cfg.getParameter<edm::InputTag>("src");
  jets_token = consumes<edm::View<pat::Jet> >(srcjets_);
  
  produces<edm::ValueMap<float> >("puppiMultiplicity");
  produces<edm::ValueMap<float> >("neutralPuppiMultiplicity");
  produces<edm::ValueMap<float> >("neutralHadronPuppiMultiplicity");
  produces<edm::ValueMap<float> >("photonPuppiMultiplicity");
  produces<edm::ValueMap<float> >("HFHadronPuppiMultiplicity");
  produces<edm::ValueMap<float> >("HFEMPuppiMultiplicity");
}

void PATPuppiJetSpecificProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const
{
  edm::Handle<edm::View<pat::Jet> > jets;
  evt.getByToken(jets_token, jets);
  
  std::vector<float> puppiMultiplicities;
  std::vector<float> neutralPuppiMultiplicities;
  std::vector<float> neutralHadronPuppiMultiplicities;
  std::vector<float> photonPuppiMultiplicities;
  std::vector<float> HFHadronPuppiMultiplicities;
  std::vector<float> HFEMPuppiMultiplicities;

  for( auto const& c : *jets ) {
    float puppiMultiplicity = 0;
    float neutralPuppiMultiplicity = 0;
    float neutralHadronPuppiMultiplicity = 0;
    float photonPuppiMultiplicity = 0;
    float HFHadronPuppiMultiplicity = 0;
    float HFEMPuppiMultiplicity = 0;

    for (unsigned i = 0; i < c.numberOfDaughters(); i++) {
        const pat::PackedCandidate &dau = static_cast<const pat::PackedCandidate &>(*c.daughter(i));
        auto weight = dau.puppiWeight();
        puppiMultiplicity += weight;
        // This logic is taken from RecoJets/JetProducers/src/JetSpecific.cc
        switch (std::abs(dau.pdgId())) {
          case 130: //PFCandidate::h0 :    // neutral hadron
            neutralHadronPuppiMultiplicity += weight;
            neutralPuppiMultiplicity += weight;
            break;
          case 22: //PFCandidate::gamma:   // photon
            photonPuppiMultiplicity += weight;
            neutralPuppiMultiplicity += weight;
            break;
          case 1: // PFCandidate::h_HF :    // hadron in HF
            HFHadronPuppiMultiplicity += weight;
            neutralPuppiMultiplicity += weight;
            break;
          case 2: //PFCandidate::egamma_HF :    // electromagnetic in HF
            HFEMPuppiMultiplicity += weight;
            neutralPuppiMultiplicity += weight;
            break;
        }
    }

    puppiMultiplicities.push_back(puppiMultiplicity);
    neutralPuppiMultiplicities.push_back(neutralPuppiMultiplicity);
    neutralHadronPuppiMultiplicities.push_back(neutralHadronPuppiMultiplicity);
    photonPuppiMultiplicities.push_back(photonPuppiMultiplicity);
    HFHadronPuppiMultiplicities.push_back(HFHadronPuppiMultiplicity);
    HFEMPuppiMultiplicities.push_back(HFEMPuppiMultiplicity);
  }

  std::unique_ptr<edm::ValueMap<float> > puppiMultiplicities_out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler puppiMultiplicities_filler(*puppiMultiplicities_out);
  puppiMultiplicities_filler.insert(jets,puppiMultiplicities.begin(),puppiMultiplicities.end());
  puppiMultiplicities_filler.fill();
  evt.put(std::move(puppiMultiplicities_out),"puppiMultiplicity");

  std::unique_ptr<edm::ValueMap<float> > neutralPuppiMultiplicities_out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler neutralPuppiMultiplicities_filler(*neutralPuppiMultiplicities_out);
  neutralPuppiMultiplicities_filler.insert(jets,neutralPuppiMultiplicities.begin(),neutralPuppiMultiplicities.end());
  neutralPuppiMultiplicities_filler.fill();
  evt.put(std::move(neutralPuppiMultiplicities_out),"neutralPuppiMultiplicity");

  std::unique_ptr<edm::ValueMap<float> > neutralHadronPuppiMultiplicities_out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler neutralHadronPuppiMultiplicities_filler(*neutralHadronPuppiMultiplicities_out);
  neutralHadronPuppiMultiplicities_filler.insert(jets,neutralHadronPuppiMultiplicities.begin(),neutralHadronPuppiMultiplicities.end());
  neutralHadronPuppiMultiplicities_filler.fill();
  evt.put(std::move(neutralHadronPuppiMultiplicities_out),"neutralHadronPuppiMultiplicity");

  std::unique_ptr<edm::ValueMap<float> > photonPuppiMultiplicities_out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler photonPuppiMultiplicities_filler(*photonPuppiMultiplicities_out);
  photonPuppiMultiplicities_filler.insert(jets,photonPuppiMultiplicities.begin(),photonPuppiMultiplicities.end());
  photonPuppiMultiplicities_filler.fill();
  evt.put(std::move(photonPuppiMultiplicities_out),"photonPuppiMultiplicity");

  std::unique_ptr<edm::ValueMap<float> > HFHadronPuppiMultiplicities_out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler HFHadronPuppiMultiplicities_filler(*HFHadronPuppiMultiplicities_out);
  HFHadronPuppiMultiplicities_filler.insert(jets,HFHadronPuppiMultiplicities.begin(),HFHadronPuppiMultiplicities.end());
  HFHadronPuppiMultiplicities_filler.fill();
  evt.put(std::move(HFHadronPuppiMultiplicities_out),"HFHadronPuppiMultiplicity");

  std::unique_ptr<edm::ValueMap<float> > HFEMPuppiMultiplicities_out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler HFEMPuppiMultiplicities_filler(*HFEMPuppiMultiplicities_out);
  HFEMPuppiMultiplicities_filler.insert(jets,HFEMPuppiMultiplicities.begin(),HFEMPuppiMultiplicities.end());
  HFEMPuppiMultiplicities_filler.fill();
  evt.put(std::move(HFEMPuppiMultiplicities_out),"HFEMPuppiMultiplicity");
}

void PATPuppiJetSpecificProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
   edm::ParameterSetDescription desc;
   desc.add<edm::InputTag>("src", edm::InputTag("slimmedJets"));
   descriptions.add("patPuppiJetSpecificProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATPuppiJetSpecificProducer);
