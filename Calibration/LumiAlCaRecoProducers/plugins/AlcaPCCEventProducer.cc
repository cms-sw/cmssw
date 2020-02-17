/**_________________________________________________________________
class:   AlcaPCCEventProducer.cc



authors: Sam Higginbotham (shigginb@cern.ch), Chris Palmer (capalmer@cern.ch), Attila Radl (attila.radl@cern.ch)

________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/Luminosity/interface/PixelClusterCountsPerEvent.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "TMath.h"
//The class
class AlcaPCCEventProducer : public edm::stream::EDProducer<> {
public:
  explicit AlcaPCCEventProducer(const edm::ParameterSet&);
  ~AlcaPCCEventProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelToken;
  edm::InputTag fPixelClusterLabel;

  std::string trigstring_;  //specifies the trigger Rand or ZeroBias
  int countEvt_;            //counter
  int countLumi_;           //counter

  std::unique_ptr<reco::PixelClusterCountsPerEvent> thePCCob;
};

//--------------------------------------------------------------------------------------------------
AlcaPCCEventProducer::AlcaPCCEventProducer(const edm::ParameterSet& iConfig) {
  fPixelClusterLabel = iConfig.getParameter<edm::ParameterSet>("AlcaPCCEventProducerParameters")
                           .getParameter<edm::InputTag>("pixelClusterLabel");
  trigstring_ = iConfig.getParameter<edm::ParameterSet>("AlcaPCCEventProducerParameters")
                    .getUntrackedParameter<std::string>("trigstring", "alcaPCCEvent");

  produces<reco::PixelClusterCountsPerEvent, edm::Transition::Event>(trigstring_);
  pixelToken = consumes<edmNew::DetSetVector<SiPixelCluster> >(fPixelClusterLabel);
}

//--------------------------------------------------------------------------------------------------
AlcaPCCEventProducer::~AlcaPCCEventProducer() {}

//--------------------------------------------------------------------------------------------------
void AlcaPCCEventProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  countEvt_++;
  thePCCob = std::make_unique<reco::PixelClusterCountsPerEvent>();

  unsigned int bx = iEvent.bunchCrossing();

  //Looping over the clusters and adding the counts up
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
  iEvent.getByToken(pixelToken, hClusterColl);

  const edmNew::DetSetVector<SiPixelCluster>& clustColl = *(hClusterColl.product());
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
    thePCCob->increment(detId(), nCluster);
    thePCCob->setbxID(bx);
  }

  iEvent.put(std::move(thePCCob), std::string(trigstring_));
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCEventProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription evtParamDesc;
  evtParamDesc.add<edm::InputTag>("pixelClusterLabel");
  evtParamDesc.addUntracked<std::string>("trigstring","alcaPCCEvent");
  desc.add<edm::ParameterSetDescription>("AlcaPCCEventProducerParameters",evtParamDesc);
  descriptions.add("alcaPCCEventProducer", desc);
}


DEFINE_FWK_MODULE(AlcaPCCEventProducer);
