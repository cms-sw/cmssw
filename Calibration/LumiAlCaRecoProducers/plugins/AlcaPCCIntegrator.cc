/*_________________________________________________________________
class:   AlcaPCCIntegrator.cc



authors: Sam Higginbotham (shigginb@cern.ch), Chris Palmer (capalmer@cern.ch), Attila Radl (attila.radl@cern.ch) 

________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "DataFormats/Luminosity/interface/PixelClusterCounts.h"
#include "DataFormats/Luminosity/interface/PixelClusterCountsInEvent.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
//The class
class AlcaPCCIntegrator
    : public edm::one::EDProducer<edm::EndLuminosityBlockProducer, edm::one::WatchLuminosityBlocks> {
public:
  explicit AlcaPCCIntegrator(const edm::ParameterSet&);
  ~AlcaPCCIntegrator() override = default;

private:
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) override;
  void endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<reco::PixelClusterCountsInEvent> pccToken_;
  std::string pccSource_;

  std::string trigstring_;  //specifies the input trigger Rand or ZeroBias
  std::string prodInst_;    //file product instance
  int countEvt_;            //counter
  int countLumi_;           //counter

  std::unique_ptr<reco::PixelClusterCounts> thePCCob;
};

//--------------------------------------------------------------------------------------------------
AlcaPCCIntegrator::AlcaPCCIntegrator(const edm::ParameterSet& iConfig) {
  pccSource_ =
      iConfig.getParameter<edm::ParameterSet>("AlcaPCCIntegratorParameters").getParameter<std::string>("inputPccLabel");
  auto trigstring_ = iConfig.getParameter<edm::ParameterSet>("AlcaPCCIntegratorParameters")
                         .getUntrackedParameter<std::string>("trigstring", "alcaPCC");
  prodInst_ =
      iConfig.getParameter<edm::ParameterSet>("AlcaPCCIntegratorParameters").getParameter<std::string>("ProdInst");

  edm::InputTag PCCInputTag_(pccSource_, trigstring_);

  countLumi_ = 0;

  produces<reco::PixelClusterCounts, edm::Transition::EndLuminosityBlock>(prodInst_);
  pccToken_ = consumes<reco::PixelClusterCountsInEvent>(PCCInputTag_);
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCIntegrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  countEvt_++;

  unsigned int bx = iEvent.bunchCrossing();
  //std::cout<<"The Bunch Crossing Int"<<bx<<std::endl;

  thePCCob->eventCounter(bx);

  //Looping over the clusters and adding the counts up
  edm::Handle<reco::PixelClusterCountsInEvent> pccHandle;
  iEvent.getByToken(pccToken_, pccHandle);

  if (!pccHandle.isValid()) {
    // do not resolve a not existing product!
    return;
  }

  const reco::PixelClusterCountsInEvent inputPcc = *pccHandle;
  thePCCob->add(inputPcc);
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCIntegrator::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {
  //PCC object at the beginning of each lumi section
  thePCCob = std::make_unique<reco::PixelClusterCounts>();
  countLumi_++;
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCIntegrator::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {}

//--------------------------------------------------------------------------------------------------
void AlcaPCCIntegrator::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) {
  //Saving the PCC object
  lumiSeg.put(std::move(thePCCob), std::string(prodInst_));
}

DEFINE_FWK_MODULE(AlcaPCCIntegrator);
