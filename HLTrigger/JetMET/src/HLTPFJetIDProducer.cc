#include "HLTrigger/JetMET/interface/HLTPFJetIDProducer.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

HLTPFJetIDProducer::HLTPFJetIDProducer(const edm::ParameterSet& iConfig) :
  jetsInput_   (iConfig.getParameter<edm::InputTag>("jetsInput")),
  min_NHEF_     (iConfig.getParameter<double>("min_NHEF")),
  max_NHEF_     (iConfig.getParameter<double>("max_NHEF")),
  min_NEMF_     (iConfig.getParameter<double>("min_NEMF")),
  max_NEMF_     (iConfig.getParameter<double>("max_NEMF")),
  min_CEMF_     (iConfig.getParameter<double>("min_CEMF")),
  max_CEMF_     (iConfig.getParameter<double>("max_CEMF")),
  min_CHEF_     (iConfig.getParameter<double>("min_CHEF")),
  max_CHEF_     (iConfig.getParameter<double>("max_CHEF")),
  min_pt_       (iConfig.getParameter<double>("min_pt"))
{
  produces< reco::PFJetCollection > ();
}

void HLTPFJetIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsInput",edm::InputTag("hltAntiKT5PFJets"));
  desc.add<double>("min_NHEF",-999.0);
  desc.add<double>("max_NHEF",999.0);
  desc.add<double>("min_NEMF",-999.0);
  desc.add<double>("max_NEMF",999.0);
  desc.add<double>("min_CEMF",-999.0);
  desc.add<double>("max_CEMF",999.0);
  desc.add<double>("min_CHEF",-999.0);
  desc.add<double>("max_CHEF",999.0);
  desc.add<double>("min_pt",30.0);
  descriptions.add("hltPFJetIDProducer", desc);
}

void HLTPFJetIDProducer::beginJob()
{

}

HLTPFJetIDProducer::~HLTPFJetIDProducer()
{

}

void HLTPFJetIDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<reco::PFJetCollection> pfjets;
  iEvent.getByLabel(jetsInput_, pfjets);

  std::auto_ptr<reco::PFJetCollection> result (new reco::PFJetCollection);

  for (reco::PFJetCollection::const_iterator pfjetc = pfjets->begin(); 
       pfjetc != pfjets->end(); ++pfjetc) {
      
    if (std::abs(pfjetc->eta()) >= 2.4) {
      result->push_back(*pfjetc);
    } else {
      if (pfjetc->pt()<min_pt_) result->push_back(*pfjetc);
      else if ((pfjetc->neutralHadronEnergyFraction() >= min_NHEF_) && (pfjetc->neutralHadronEnergyFraction() <= max_NHEF_) && 
	       (pfjetc->neutralEmEnergyFraction() >= min_NEMF_) && (pfjetc->neutralEmEnergyFraction() <= max_NEMF_) &&
	       (pfjetc->chargedEmEnergyFraction() >= min_CEMF_) && (pfjetc->chargedEmEnergyFraction() <= max_CEMF_) &&
	       (pfjetc->chargedHadronEnergyFraction() >= min_CHEF_) && (pfjetc->chargedHadronEnergyFraction() <= max_CHEF_)) {
	result->push_back(*pfjetc);
      }
    }
  } // pfjetc

  iEvent.put( result);

}
