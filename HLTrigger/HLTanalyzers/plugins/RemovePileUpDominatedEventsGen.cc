// -*- C++ -*-
//
// Package:    RemovePileUpDominatedEventsGen/RemovePileUpDominatedEventsGen
// Class:      RemovePileUpDominatedEventsGen
//
/**\class RemovePileUpDominatedEventsGen RemovePileUpDominatedEventsGen.cc RemovePileUpDominatedEventsGen/RemovePileUpDominatedEventsGen/plugins/RemovePileUpDominatedEventsGen.cc

 Description: [one line class summary]

This EDFilter select events having the generator pt-hat greater than the pt-had of the pile-up collisions.
This code is used by STEAM-TSG in order to estimate the HLT rate using MC sample, especially to avoid a rate double counting due to pile-up.
For more information on the theory of this method see Appendix B of https://www.dropbox.com/home/TesiPhD_Silvio?preview=PhD_thesis.pdf .  


 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Silvio DONATO
//         Created:  Fri, 12 Dec 2014 12:48:57 GMT
//
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <iostream>
#include <SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h>
#include <SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h>

class RemovePileUpDominatedEventsGen : public edm::stream::EDFilter<> {
public:
  explicit RemovePileUpDominatedEventsGen(const edm::ParameterSet&);
  ~RemovePileUpDominatedEventsGen() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileupSummaryInfos_;
  const edm::EDGetTokenT<GenEventInfoProduct> generatorInfo_;
  unsigned int bunchCrossing;
};

RemovePileUpDominatedEventsGen::RemovePileUpDominatedEventsGen(const edm::ParameterSet& iConfig)
    : pileupSummaryInfos_(
          consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter<edm::InputTag>("pileupSummaryInfos"))),
      generatorInfo_(consumes<GenEventInfoProduct>(iConfig.getParameter<edm::InputTag>("generatorInfo"))) {
  bunchCrossing = 0;
  produces<float>();
}

RemovePileUpDominatedEventsGen::~RemovePileUpDominatedEventsGen() = default;

bool RemovePileUpDominatedEventsGen::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  edm::Handle<GenEventInfoProduct> generatorInfo;
  iEvent.getByToken(generatorInfo_, generatorInfo);

  edm::Handle<std::vector<PileupSummaryInfo> > pileupSummaryInfos;
  iEvent.getByToken(pileupSummaryInfos_, pileupSummaryInfos);

  //find in-time pile-up
  if (bunchCrossing >= pileupSummaryInfos.product()->size() ||
      pileupSummaryInfos.product()->at(bunchCrossing).getBunchCrossing() != 0) {
    bool found = false;
    for (bunchCrossing = 0; bunchCrossing < pileupSummaryInfos.product()->size() && !found; ++bunchCrossing) {
      if (pileupSummaryInfos.product()->at(bunchCrossing).getBunchCrossing() == 0) {
        found = true;
        bunchCrossing--;
      }
    }
    if (!found) {
      edm::LogInfo("RemovePileUpDominatedEventsGen") << "In-time pile-up not found!" << endl;
      return true;
    }
  }

  //   cout << "Using "<<bunchCrossing<<endl;
  //   cout << "pileupSummaryInfos.product()->at(bunchCrossing).getBunchCrossing() "<<pileupSummaryInfos.product()->at(bunchCrossing).getBunchCrossing()<<endl;

  //get the PU pt-hat max
  float signal_pT_hat = -1;
  float pu_pT_hat_max = -1;

  PileupSummaryInfo puSummary_onTime = pileupSummaryInfos.product()->at(bunchCrossing);
  for (const auto& pu_pT_hat : puSummary_onTime.getPU_pT_hats())
    if (pu_pT_hat > pu_pT_hat_max)
      pu_pT_hat_max = pu_pT_hat;

  //get the signal pt-hat
  signal_pT_hat = generatorInfo->qScale();

  //save PU - signal pt-hat
  std::unique_ptr<float> pOut(new float());
  *pOut = signal_pT_hat - pu_pT_hat_max;
  iEvent.put(std::move(pOut));

  //filter the event
  if (signal_pT_hat > pu_pT_hat_max)
    return true;
  return false;
}

void RemovePileUpDominatedEventsGen::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pileupSummaryInfos", edm::InputTag("addPileupInfo"));
  desc.add<edm::InputTag>("generatorInfo", edm::InputTag("generator"));
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(RemovePileUpDominatedEventsGen);
