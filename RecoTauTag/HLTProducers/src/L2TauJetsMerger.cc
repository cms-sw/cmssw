#include "RecoTauTag/HLTProducers/interface/L2TauJetsMerger.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;

L2TauJetsMerger::L2TauJetsMerger(const edm::ParameterSet& iConfig)
    : jetSrc(iConfig.getParameter<vtag>("JetSrc")), mEt_Min(iConfig.getParameter<double>("EtMin")) {
  for (vtag::const_iterator it = jetSrc.begin(); it != jetSrc.end(); ++it) {
    edm::EDGetTokenT<CaloJetCollection> aToken = consumes<CaloJetCollection>(*it);
    jetSrc_token.push_back(aToken);
  }

  produces<CaloJetCollection>();
}

L2TauJetsMerger::~L2TauJetsMerger() {}

void L2TauJetsMerger::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const {
  using namespace edm;
  using namespace std;
  using namespace reco;

  //Getting the Collections of L2ReconstructedJets from L1Seeds
  //and removing the collinear jets
  CaloJetCollection myTmpJets;

  int iL1Jet = 0;
  for (vtoken_cjets::const_iterator s = jetSrc_token.begin(); s != jetSrc_token.end(); ++s) {
    edm::Handle<CaloJetCollection> tauJets;
    iEvent.getByToken(*s, tauJets);
    for (CaloJetCollection::const_iterator iTau = tauJets->begin(); iTau != tauJets->end(); ++iTau) {
      if (iTau->et() > mEt_Min) {
        //Add the Pdg Id here
        CaloJet myJet = *iTau;
        myJet.setPdgId(15);
        myTmpJets.push_back(myJet);
      }
    }
    iL1Jet++;
  }

  std::unique_ptr<CaloJetCollection> tauL2jets(new CaloJetCollection);

  //Removing the collinear jets correctly!

  //First sort the jets you have merged
  SorterByPt sorter;
  std::sort(myTmpJets.begin(), myTmpJets.end(), sorter);

  //Remove Collinear Jets by prefering the highest ones!
  while (!myTmpJets.empty()) {
    tauL2jets->push_back(myTmpJets[0]);
    CaloJetCollection tmp;
    for (unsigned int i = 1; i < myTmpJets.size(); ++i) {
      double DR2 = ROOT::Math::VectorUtil::DeltaR2(myTmpJets[0].p4(), myTmpJets[i].p4());
      if (DR2 > 0.1 * 0.1)
        tmp.push_back(myTmpJets[i]);
    }
    myTmpJets.swap(tmp);
  }

  iEvent.put(std::move(tauL2jets));
}

void L2TauJetsMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<edm::InputTag> inputTags;
  inputTags.push_back(edm::InputTag("hltAkIsoTau1Regional"));
  inputTags.push_back(edm::InputTag("hltAkIsoTau2Regional"));
  inputTags.push_back(edm::InputTag("hltAkIsoTau3Regional"));
  inputTags.push_back(edm::InputTag("hltAkIsoTau4Regional"));
  desc.add<std::vector<edm::InputTag> >("JetSrc", inputTags)->setComment("CaloJet collections to merge");
  desc.add<double>("EtMin", 20.0)->setComment("Minimal ET of jet to merge");
  descriptions.setComment("Merges CaloJet collections removing duplicates");
  descriptions.add("L2TauJetsMerger", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L2TauJetsMerger);
