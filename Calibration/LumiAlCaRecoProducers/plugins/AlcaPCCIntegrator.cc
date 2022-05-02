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
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
//The class
class AlcaPCCIntegrator : public edm::global::EDProducer<edm::LuminosityBlockSummaryCache<reco::PixelClusterCounts>,
                                                         edm::StreamCache<reco::PixelClusterCounts>,
                                                         edm::EndLuminosityBlockProducer,
                                                         edm::Accumulator> {
public:
  explicit AlcaPCCIntegrator(const edm::ParameterSet&);
  ~AlcaPCCIntegrator() override = default;

  std::unique_ptr<reco::PixelClusterCounts> beginStream(edm::StreamID) const override;
  std::shared_ptr<reco::PixelClusterCounts> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                              edm::EventSetup const&) const override;
  void accumulate(edm::StreamID iID, const edm::Event& iEvent, const edm::EventSetup&) const override;
  void streamEndLuminosityBlockSummary(edm::StreamID,
                                       edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       reco::PixelClusterCounts*) const override;
  void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       reco::PixelClusterCounts* iCounts) const override;
  void globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLumi,
                                       edm::EventSetup const&,
                                       reco::PixelClusterCounts const* iCounts) const override;

private:
  edm::EDPutTokenT<reco::PixelClusterCounts> lumiPutToken_;
  edm::InputTag thePCCInputTag_;
  edm::EDGetTokenT<reco::PixelClusterCountsInEvent> pccToken_;
};

AlcaPCCIntegrator::AlcaPCCIntegrator(const edm::ParameterSet& iConfig)
    : lumiPutToken_(produces<reco::PixelClusterCounts, edm::Transition::EndLuminosityBlock>(
          iConfig.getParameter<edm::ParameterSet>("AlcaPCCIntegratorParameters").getParameter<std::string>("ProdInst"))),
      thePCCInputTag_(iConfig.getParameter<edm::ParameterSet>("AlcaPCCIntegratorParameters")
                          .getParameter<std::string>("inputPccLabel"),
                      iConfig.getParameter<edm::ParameterSet>("AlcaPCCIntegratorParameters")
                          .getUntrackedParameter<std::string>("trigstring", "alcaPCC")),
      pccToken_(consumes<reco::PixelClusterCountsInEvent>(thePCCInputTag_)) {}

std::unique_ptr<reco::PixelClusterCounts> AlcaPCCIntegrator::beginStream(edm::StreamID StreamID) const {
  return std::make_unique<reco::PixelClusterCounts>();
}

std::shared_ptr<reco::PixelClusterCounts> AlcaPCCIntegrator::globalBeginLuminosityBlockSummary(
    edm::LuminosityBlock const&, edm::EventSetup const&) const {
  return std::make_shared<reco::PixelClusterCounts>();
}

void AlcaPCCIntegrator::globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                        edm::EventSetup const&,
                                                        reco::PixelClusterCounts*) const {}

void AlcaPCCIntegrator::accumulate(edm::StreamID iID, const edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<reco::PixelClusterCountsInEvent> pccHandle;
  iEvent.getByToken(pccToken_, pccHandle);

  if (!pccHandle.isValid()) {
    // do not resolve a not existing product!
    return;
  }

  const reco::PixelClusterCountsInEvent inputPcc = *pccHandle;
  unsigned int bx = iEvent.bunchCrossing();
  // add the BXID of the event to the stream cache
  streamCache(iID)->eventCounter(bx);
  // add the PCCs from the event to the stream cache
  streamCache(iID)->add(inputPcc);
}

void AlcaPCCIntegrator::streamEndLuminosityBlockSummary(edm::StreamID iID,
                                                        edm::LuminosityBlock const&,
                                                        edm::EventSetup const&,
                                                        reco::PixelClusterCounts* iCounts) const {
  iCounts->merge(*streamCache(iID));
  // now clear in order to be ready for the next LuminosityBlock
  streamCache(iID)->reset();
}

void AlcaPCCIntegrator::globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLumi,
                                                        edm::EventSetup const&,
                                                        reco::PixelClusterCounts const* iCounts) const {
  // save the PCC object
  iLumi.emplace(lumiPutToken_, *iCounts);
}
DEFINE_FWK_MODULE(AlcaPCCIntegrator);
