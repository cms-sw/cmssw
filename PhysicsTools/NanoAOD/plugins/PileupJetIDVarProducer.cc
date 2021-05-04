#include <memory>
#include <vector>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"

class PileupJetIDVarProducer : public edm::global::EDProducer<> {
public:
  explicit PileupJetIDVarProducer(const edm::ParameterSet& iConfig)
      : srcJet_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("srcJet"))),
        srcPileupJetId_(
            consumes<edm::ValueMap<StoredPileupJetIdentifier>>(iConfig.getParameter<edm::InputTag>("srcPileupJetId"))) {
    produces<edm::ValueMap<float>>("dR2Mean");
    produces<edm::ValueMap<float>>("majW");
    produces<edm::ValueMap<float>>("minW");
    produces<edm::ValueMap<float>>("frac01");
    produces<edm::ValueMap<float>>("frac02");
    produces<edm::ValueMap<float>>("frac03");
    produces<edm::ValueMap<float>>("frac04");
    produces<edm::ValueMap<float>>("ptD");
    produces<edm::ValueMap<float>>("beta");
    produces<edm::ValueMap<float>>("pull");
    produces<edm::ValueMap<float>>("jetR");
    produces<edm::ValueMap<float>>("jetRchg");
    produces<edm::ValueMap<int>>("nCharged");
  }
  ~PileupJetIDVarProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<pat::Jet>> srcJet_;
  edm::EDGetTokenT<edm::ValueMap<StoredPileupJetIdentifier>> srcPileupJetId_;
};

// ------------ method called to produce the data  ------------
void PileupJetIDVarProducer::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<edm::View<pat::Jet>> srcJet;
  edm::Handle<edm::ValueMap<StoredPileupJetIdentifier>> srcPileupJetId;

  iEvent.getByToken(srcJet_, srcJet);
  iEvent.getByToken(srcPileupJetId_, srcPileupJetId);

  unsigned int nJet = srcJet->size();

  std::vector<float> dR2Mean(nJet, -1);
  std::vector<float> majW(nJet, -1);
  std::vector<float> minW(nJet, -1);
  std::vector<float> frac01(nJet, -1);
  std::vector<float> frac02(nJet, -1);
  std::vector<float> frac03(nJet, -1);
  std::vector<float> frac04(nJet, -1);
  std::vector<float> ptD(nJet, -1);
  std::vector<float> beta(nJet, -1);
  std::vector<float> pull(nJet, -1);
  std::vector<float> jetR(nJet, -1);
  std::vector<float> jetRchg(nJet, -1);
  std::vector<int> nCharged(nJet, -1);

  for (unsigned int ij = 0; ij < nJet; ij++) {
    auto jet = srcJet->ptrAt(ij);

    edm::RefToBase<pat::Jet> jetRef = srcJet->refAt(ij);

    dR2Mean[ij] = (*srcPileupJetId)[jetRef].dR2Mean();
    majW[ij] = (*srcPileupJetId)[jetRef].majW();
    minW[ij] = (*srcPileupJetId)[jetRef].minW();
    frac01[ij] = (*srcPileupJetId)[jetRef].frac01();
    frac02[ij] = (*srcPileupJetId)[jetRef].frac02();
    frac03[ij] = (*srcPileupJetId)[jetRef].frac03();
    frac04[ij] = (*srcPileupJetId)[jetRef].frac04();
    ptD[ij] = (*srcPileupJetId)[jetRef].ptD();
    beta[ij] = (*srcPileupJetId)[jetRef].beta();
    pull[ij] = (*srcPileupJetId)[jetRef].pull();
    jetR[ij] = (*srcPileupJetId)[jetRef].jetR();
    jetRchg[ij] = (*srcPileupJetId)[jetRef].jetRchg();
    nCharged[ij] = (*srcPileupJetId)[jetRef].nCharged();
  }

  std::unique_ptr<edm::ValueMap<float>> dR2MeanV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_dR2Mean(*dR2MeanV);
  filler_dR2Mean.insert(srcJet, dR2Mean.begin(), dR2Mean.end());
  filler_dR2Mean.fill();
  iEvent.put(std::move(dR2MeanV), "dR2Mean");

  std::unique_ptr<edm::ValueMap<float>> majWV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_majW(*majWV);
  filler_majW.insert(srcJet, majW.begin(), majW.end());
  filler_majW.fill();
  iEvent.put(std::move(majWV), "majW");

  std::unique_ptr<edm::ValueMap<float>> minWV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_minW(*minWV);
  filler_minW.insert(srcJet, minW.begin(), minW.end());
  filler_minW.fill();
  iEvent.put(std::move(minWV), "minW");

  std::unique_ptr<edm::ValueMap<float>> frac01V(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_frac01(*frac01V);
  filler_frac01.insert(srcJet, frac01.begin(), frac01.end());
  filler_frac01.fill();
  iEvent.put(std::move(frac01V), "frac01");

  std::unique_ptr<edm::ValueMap<float>> frac02V(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_frac02(*frac02V);
  filler_frac02.insert(srcJet, frac02.begin(), frac02.end());
  filler_frac02.fill();
  iEvent.put(std::move(frac02V), "frac02");

  std::unique_ptr<edm::ValueMap<float>> frac03V(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_frac03(*frac03V);
  filler_frac03.insert(srcJet, frac03.begin(), frac03.end());
  filler_frac03.fill();
  iEvent.put(std::move(frac03V), "frac03");

  std::unique_ptr<edm::ValueMap<float>> frac04V(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_frac04(*frac04V);
  filler_frac04.insert(srcJet, frac04.begin(), frac04.end());
  filler_frac04.fill();
  iEvent.put(std::move(frac04V), "frac04");

  std::unique_ptr<edm::ValueMap<float>> ptDV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_ptD(*ptDV);
  filler_ptD.insert(srcJet, ptD.begin(), ptD.end());
  filler_ptD.fill();
  iEvent.put(std::move(ptDV), "ptD");

  std::unique_ptr<edm::ValueMap<float>> betaV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_beta(*betaV);
  filler_beta.insert(srcJet, beta.begin(), beta.end());
  filler_beta.fill();
  iEvent.put(std::move(betaV), "beta");

  std::unique_ptr<edm::ValueMap<float>> pullV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_pull(*pullV);
  filler_pull.insert(srcJet, pull.begin(), pull.end());
  filler_pull.fill();
  iEvent.put(std::move(pullV), "pull");

  std::unique_ptr<edm::ValueMap<float>> jetRV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_jetR(*jetRV);
  filler_jetR.insert(srcJet, jetR.begin(), jetR.end());
  filler_jetR.fill();
  iEvent.put(std::move(jetRV), "jetR");

  std::unique_ptr<edm::ValueMap<float>> jetRchgV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_jetRchg(*jetRchgV);
  filler_jetRchg.insert(srcJet, jetRchg.begin(), jetRchg.end());
  filler_jetRchg.fill();
  iEvent.put(std::move(jetRchgV), "jetRchg");

  std::unique_ptr<edm::ValueMap<int>> nChargedV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filler_nCharged(*nChargedV);
  filler_nCharged.insert(srcJet, nCharged.begin(), nCharged.end());
  filler_nCharged.fill();
  iEvent.put(std::move(nChargedV), "nCharged");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PileupJetIDVarProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJet")->setComment("jet input collection");
  desc.add<edm::InputTag>("srcPileupJetId")->setComment("StoredPileupJetIdentifier name");
  std::string modname;
  modname += "PileupJetIDVarProducer";
  descriptions.add(modname, desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PileupJetIDVarProducer);
