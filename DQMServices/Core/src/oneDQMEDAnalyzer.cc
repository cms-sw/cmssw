#include "DQMServices/Core/interface/oneDQMEDAnalyzer.h"

using namespace one::dqmimplementation;


DQMLumisEDProducer::DQMLumisEDProducer():
  lumiToken_{produces<DQMToken,edm::Transition::EndLuminosityBlock>("endLumi")}
{}

void DQMLumisEDProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) 
{ 
  dqmBeginLuminosityBlock(lumi, setup);

  /*
  edm::Service<DQMStore>()->bookTransaction(
    [this, &lumi, &setup](DQMStore::IBooker & booker)
    {
      booker.cd();
      this->bookLumiHistograms(booker, lumi, setup);
    },
    lumi.run(),
    lumi.moduleCallingContext()->moduleDescription()->id());
  */
}

void DQMLumisEDProducer::dqmBeginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) 
{ }

void DQMLumisEDProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) 
{ }

void DQMLumisEDProducer::endLuminosityBlockProduce(edm::LuminosityBlock & lumi, edm::EventSetup const& setup) 
{
  edm::Service<DQMStore>()->cloneLumiHistograms(
      lumi.run(),
      lumi.luminosityBlock(),
      moduleDescription().id());

  lumi.emplace<DQMToken>(lumiToken_);
}
