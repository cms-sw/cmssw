// -*- C++ -*-
//111
// Package:    HiSpikeCleaner
// Class:      HiSpikeCleaner
//
/**\class HiSpikeCleaner HiSpikeCleaner.cc RecoHI/HiSpikeCleaner/src/HiSpikeCleaner.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yong Kim,32 4-A08,+41227673039,
//         Created:  Mon Nov  1 18:22:21 CET 2010
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

//
// class declaration
//

class HiSpikeCleaner : public edm::stream::EDProducer<> {
public:
  explicit HiSpikeCleaner(const edm::ParameterSet&);
  ~HiSpikeCleaner() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<reco::SuperClusterCollection> sCInputProducerToken_;
  edm::EDGetTokenT<EcalRecHitCollection> rHInputProducerBToken_;
  edm::EDGetTokenT<EcalRecHitCollection> rHInputProducerEToken_;
  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  std::string outputCollection_;
  double TimingCut_;
  double swissCutThr_;
  double etCut_;
};

HiSpikeCleaner::HiSpikeCleaner(const edm::ParameterSet& iConfig) : ecalClusterToolsESGetTokens_{consumesCollector()} {
  //register your products
  /* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/
  //now do what ever other initialization is needed

  rHInputProducerBToken_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitProducerBarrel"));
  rHInputProducerEToken_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitProducerEndcap"));

  sCInputProducerToken_ =
      consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("originalSuperClusterProducer"));
  TimingCut_ = iConfig.getUntrackedParameter<double>("TimingCut", 4.0);
  swissCutThr_ = iConfig.getUntrackedParameter<double>("swissCutThr", 0.95);
  etCut_ = iConfig.getParameter<double>("etCut");

  outputCollection_ = iConfig.getParameter<std::string>("outputColl");
  produces<reco::SuperClusterCollection>(outputCollection_);
}

HiSpikeCleaner::~HiSpikeCleaner() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HiSpikeCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get raw SuperClusters from the event
  Handle<reco::SuperClusterCollection> pRawSuperClusters;
  try {
    iEvent.getByToken(sCInputProducerToken_, pRawSuperClusters);
  } catch (cms::Exception& ex) {
    edm::LogError("EgammaSCCorrectionMakerError") << "Error! can't get the rawSuperClusters ";
  }

  // Get the RecHits from the event
  Handle<EcalRecHitCollection> pRecHitsB;
  try {
    iEvent.getByToken(rHInputProducerBToken_, pRecHitsB);
  } catch (cms::Exception& ex) {
    edm::LogError("EgammaSCCorrectionMakerError") << "Error! can't get the RecHits ";
  }

  // Get the RecHits from the event
  Handle<EcalRecHitCollection> pRecHitsE;
  try {
    iEvent.getByToken(rHInputProducerEToken_, pRecHitsE);
  } catch (cms::Exception& ex) {
    edm::LogError("EgammaSCCorrectionMakerError") << "Error! can't get the RecHits ";
  }

  // get the channel status from the DB
  //   edm::ESHandle<EcalChannelStatus> chStatus;
  //   iSetup.get<EcalChannelStatusRcd>().get(chStatus);

  edm::ESHandle<EcalSeverityLevelAlgo> ecalSevLvlAlgoHndl;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(ecalSevLvlAlgoHndl);

  // Create a pointer to the RecHits and raw SuperClusters
  const reco::SuperClusterCollection* rawClusters = pRawSuperClusters.product();

  EcalClusterLazyTools lazyTool(
      iEvent, ecalClusterToolsESGetTokens_.get(iSetup), rHInputProducerBToken_, rHInputProducerEToken_);

  // Define a collection of corrected SuperClusters to put back into the event
  auto corrClusters = std::make_unique<reco::SuperClusterCollection>();

  //  Loop over raw clusters and make corrected ones
  reco::SuperClusterCollection::const_iterator aClus;
  for (aClus = rawClusters->begin(); aClus != rawClusters->end(); aClus++) {
    double theEt = aClus->energy() / cosh(aClus->eta());
    //	 std::cout << " et of SC = " << theEt << std::endl;

    if (theEt < etCut_)
      continue;  // cut off low pT superclusters

    bool flagS = true;
    float swissCrx(0);

    const reco::CaloClusterPtr seed = aClus->seed();
    DetId id = lazyTool.getMaximum(*seed).first;
    const EcalRecHitCollection& rechits = *pRecHitsB;
    EcalRecHitCollection::const_iterator it = rechits.find(id);

    if (it != rechits.end()) {
      ecalSevLvlAlgoHndl->severityLevel(id, rechits);
      swissCrx = EcalTools::swissCross(id, rechits, 0., true);
      //	    std::cout << "swissCross = " << swissCrx <<std::endl;
      // std::cout << " timing = " << it->time() << std::endl;
    }

    if (fabs(it->time()) > TimingCut_) {
      flagS = false;
      //	    std::cout << " timing = " << it->time() << std::endl;
      //   std::cout << " timing is bad........" << std::endl;
    }
    if (swissCrx > (float)swissCutThr_) {
      flagS = false;  // swissCross cut
                      //	    std::cout << "swissCross = " << swissCrx <<std::endl;
                      //   std::cout << " removed by swiss cross cut" << std::endl;
    }
    // - kGood        --> good channel
    // - kProblematic --> problematic (e.g. noisy)
    // - kRecovered   --> recovered (e.g. an originally dead or saturated)
    // - kTime        --> the channel is out of time (e.g. spike)
    // - kWeird       --> weird (e.g. spike)
    // - kBad         --> bad, not suitable to be used in the reconstruction
    //   enum EcalSeverityLevel { kGood=0, kProblematic, kRecovered, kTime, kWeird, kBad };

    reco::SuperCluster newClus;
    if (flagS == true)
      newClus = *aClus;
    else
      continue;
    corrClusters->push_back(newClus);
  }

  // Put collection of corrected SuperClusters into the event
  iEvent.put(std::move(corrClusters), outputCollection_);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiSpikeCleaner);
