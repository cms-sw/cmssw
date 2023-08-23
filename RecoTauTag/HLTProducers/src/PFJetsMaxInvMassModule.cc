#include "RecoTauTag/HLTProducers/interface/PFJetsMaxInvMassModule.h"
#include "Math/GenVector/VectorUtil.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"

//
// class declaration
//
PFJetsMaxInvMassModule::PFJetsMaxInvMassModule(const edm::ParameterSet& iConfig)
    : pfJetSrc_(consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("PFJetSrc"))),
      maxInvMassPairOnly_(iConfig.getParameter<bool>("maxInvMassPairOnly")),
      removeMaxInvMassPair_(iConfig.getParameter<bool>("removeMaxInvMassPair")) {
  produces<reco::PFJetCollection>();
}

void PFJetsMaxInvMassModule::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const {
  std::unique_ptr<reco::PFJetCollection> addPFJets(new reco::PFJetCollection);

  edm::Handle<reco::PFJetCollection> jets;
  iEvent.getByToken(pfJetSrc_, jets);

  unsigned iCan = 0;
  unsigned jCan = 0;
  double m2jj_max = 0;

  if (jets->size() > 1) {
    for (unsigned i = 0; i < jets->size(); i++) {
      for (unsigned j = i + 1; j < jets->size(); j++) {
        double test = ((*jets)[i].p4() + (*jets)[j].p4()).M2();
        if (test > m2jj_max) {
          m2jj_max = test;
          iCan = i;
          jCan = j;
        }
      }
    }

    if (maxInvMassPairOnly_) {
      addPFJets->push_back((*jets)[iCan]);
      addPFJets->push_back((*jets)[jCan]);
    } else if (removeMaxInvMassPair_) {
      for (unsigned i = 0; i < jets->size(); i++) {
        if (i != iCan && i != jCan)
          addPFJets->push_back((*jets)[i]);
      }
    }
  }

  iEvent.put(std::move(addPFJets));
}

void PFJetsMaxInvMassModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("PFJetSrc", edm::InputTag(""))->setComment("Input collection of PFJets that pass a filter");
  desc.add<bool>("maxInvMassPairOnly", true)->setComment("Add only max mjj pair");
  desc.add<bool>("removeMaxInvMassPair", false)->setComment("Remove max mjj pair and keep all other jets");
  descriptions.setComment(
      "This module produces a collection of PFJets that are cross-cleaned with respect to PFTaus passing a HLT "
      "filter.");
  descriptions.add("PFJetsMaxInvMassModule", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetsMaxInvMassModule);
