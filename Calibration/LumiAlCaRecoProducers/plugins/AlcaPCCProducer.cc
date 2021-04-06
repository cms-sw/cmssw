/**_________________________________________________________________
class:   AlcaPCCProducer.cc



authors:Sam Higginbotham (shigginb@cern.ch) and Chris Palmer (capalmer@cern.ch) 

________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Luminosity/interface/PixelClusterCounts.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//The class
class AlcaPCCProducer : public edm::one::EDProducer<edm::EndLuminosityBlockProducer, edm::one::WatchLuminosityBlocks> {
public:
  explicit AlcaPCCProducer(const edm::ParameterSet&);

private:
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) override;
  void endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelToken_;
  edm::EDPutTokenT<reco::PixelClusterCounts> putToken_;

  std::unique_ptr<reco::PixelClusterCounts> thePCCob;
};

//--------------------------------------------------------------------------------------------------
AlcaPCCProducer::AlcaPCCProducer(const edm::ParameterSet& iConfig)
    : pixelToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelClusterLabel"))),
      //specifies the trigger Rand or ZeroBias
      putToken_(produces<reco::PixelClusterCounts, edm::Transition::EndLuminosityBlock>(
          iConfig.getUntrackedParameter<std::string>("trigstring", "alcaPCC"))) {}

//--------------------------------------------------------------------------------------------------
void AlcaPCCProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  unsigned int bx = iEvent.bunchCrossing();
  //std::cout<<"The Bunch Crossing"<<bx<<std::endl;
  thePCCob->eventCounter(bx);

  //Looping over the clusters and adding the counts up
  const edmNew::DetSetVector<SiPixelCluster>& clustColl = iEvent.get(pixelToken_);
  // ----------------------------------------------------------------------
  // -- Clusters without tracks
  for (auto const& mod : clustColl) {
    if (mod.empty()) {
      continue;
    }
    DetId detId = mod.id();

    //--The following will be used when we make a theshold for the clusters.
    //--Keeping this for features that may be implemented later.
    // -- clusters on this det
    //edmNew::DetSet<SiPixelCluster>::const_iterator  di;
    //int nClusterCount=0;
    //for (di = mod.begin(); di != mod.end(); ++di) {
    //    nClusterCount++;
    //}
    int nCluster = mod.size();
    thePCCob->increment(detId(), bx, nCluster);
  }
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {
  //New PCC object at the beginning of each lumi section
  thePCCob = std::make_unique<reco::PixelClusterCounts>();
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) {}

//--------------------------------------------------------------------------------------------------
void AlcaPCCProducer::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) {
  //Saving the PCC object
  lumiSeg.put(putToken_, std::move(thePCCob));
}

DEFINE_FWK_MODULE(AlcaPCCProducer);
