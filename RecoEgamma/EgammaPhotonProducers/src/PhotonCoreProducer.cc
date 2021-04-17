/** \class PhotonCoreProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

// PhotonCoreProducer inherits from EDProducer, so it can be a module:
class PhotonCoreProducer : public edm::stream::EDProducer<> {
public:
  PhotonCoreProducer(const edm::ParameterSet& ps);
  ~PhotonCoreProducer() override;

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  void fillPhotonCollection(edm::Event& evt,
                            edm::EventSetup const& es,
                            const edm::Handle<reco::SuperClusterCollection>& scHandle,
                            const edm::Handle<reco::ConversionCollection>& conversionHandle,
                            const edm::Handle<reco::ElectronSeedCollection>& pixelSeeds,
                            reco::PhotonCoreCollection& outputCollection,
                            int& iSC);

  reco::ConversionRef solveAmbiguity(const edm::Handle<reco::ConversionCollection>& conversionHandle,
                                     reco::SuperClusterRef& sc);

  std::string PhotonCoreCollection_;
  edm::EDGetTokenT<reco::SuperClusterCollection> scHybridBarrelProducer_;
  edm::EDGetTokenT<reco::SuperClusterCollection> scIslandEndcapProducer_;
  edm::EDGetTokenT<reco::ConversionCollection> conversionProducer_;
  edm::EDGetTokenT<reco::ElectronSeedCollection> pixelSeedProducer_;

  double minSCEt_;
  bool validConversions_;
  edm::ParameterSet conf_;
  bool validPixelSeeds_;
  bool risolveAmbiguity_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PhotonCoreProducer);

PhotonCoreProducer::PhotonCoreProducer(const edm::ParameterSet& config)
    : conf_(config)

{
  // use onfiguration file to setup input/output collection names
  scHybridBarrelProducer_ =
      consumes<reco::SuperClusterCollection>(conf_.getParameter<edm::InputTag>("scHybridBarrelProducer"));
  scIslandEndcapProducer_ =
      consumes<reco::SuperClusterCollection>(conf_.getParameter<edm::InputTag>("scIslandEndcapProducer"));
  conversionProducer_ = consumes<reco::ConversionCollection>(conf_.getParameter<edm::InputTag>("conversionProducer"));
  PhotonCoreCollection_ = conf_.getParameter<std::string>("photonCoreCollection");
  pixelSeedProducer_ = consumes<reco::ElectronSeedCollection>(conf_.getParameter<edm::InputTag>("pixelSeedProducer"));
  minSCEt_ = conf_.getParameter<double>("minSCEt");
  risolveAmbiguity_ = conf_.getParameter<bool>("risolveConversionAmbiguity");

  // Register the product
  produces<reco::PhotonCoreCollection>(PhotonCoreCollection_);
}

PhotonCoreProducer::~PhotonCoreProducer() {}

void PhotonCoreProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  using namespace edm;
  //  nEvt_++;

  reco::PhotonCoreCollection outputPhotonCoreCollection;
  auto outputPhotonCoreCollection_p = std::make_unique<reco::PhotonCoreCollection>();

  // Get the  Barrel Super Cluster collection
  bool validBarrelSCHandle = true;
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByToken(scHybridBarrelProducer_, scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("PhotonCoreProducer") << "Error! Can't get the scHybridBarrelProducer";
    validBarrelSCHandle = false;
  }

  // Get the  Endcap Super Cluster collection
  bool validEndcapSCHandle = true;
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByToken(scIslandEndcapProducer_, scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("PhotonCoreProducer") << "Error! Can't get the scIslandEndcapProducer";
    validEndcapSCHandle = false;
  }

  ///// Get the conversion collection
  validConversions_ = true;
  edm::Handle<reco::ConversionCollection> conversionHandle;
  theEvent.getByToken(conversionProducer_, conversionHandle);
  if (!conversionHandle.isValid()) {
    //edm::LogError("PhotonCoreProducer") << "Error! Can't get the product "<< conversionProducer_.label() << "\n" ;
    validConversions_ = false;
  }

  // Get ElectronPixelSeeds
  validPixelSeeds_ = true;
  Handle<reco::ElectronSeedCollection> pixelSeedHandle;
  reco::ElectronSeedCollection pixelSeeds;
  theEvent.getByToken(pixelSeedProducer_, pixelSeedHandle);
  if (!pixelSeedHandle.isValid()) {
    validPixelSeeds_ = false;
  }
  //  if ( validPixelSeeds_) pixelSeeds = *(pixelSeedHandle.product());

  int iSC = 0;  // index in photon collection
  // Loop over barrel and endcap SC collections and fill the  photon collection
  if (validBarrelSCHandle)
    fillPhotonCollection(
        theEvent, theEventSetup, scBarrelHandle, conversionHandle, pixelSeedHandle, outputPhotonCoreCollection, iSC);
  if (validEndcapSCHandle)
    fillPhotonCollection(
        theEvent, theEventSetup, scEndcapHandle, conversionHandle, pixelSeedHandle, outputPhotonCoreCollection, iSC);

  // put the product in the event
  edm::LogInfo("PhotonCoreProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCoreCollection_p->assign(outputPhotonCoreCollection.begin(), outputPhotonCoreCollection.end());
  theEvent.put(std::move(outputPhotonCoreCollection_p), PhotonCoreCollection_);
}

void PhotonCoreProducer::fillPhotonCollection(edm::Event& evt,
                                              edm::EventSetup const& es,
                                              const edm::Handle<reco::SuperClusterCollection>& scHandle,
                                              const edm::Handle<reco::ConversionCollection>& conversionHandle,
                                              const edm::Handle<reco::ElectronSeedCollection>& pixelSeedHandle,
                                              reco::PhotonCoreCollection& outputPhotonCoreCollection,
                                              int& iSC) {
  for (unsigned int lSC = 0; lSC < scHandle->size(); lSC++) {
    // get SuperClusterRef
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scHandle, lSC));
    iSC++;
    //const reco::SuperCluster* pClus=&(*scRef);

    // SC energy preselection
    if (scRef->energy() / cosh(scRef->eta()) <= minSCEt_)
      continue;

    reco::PhotonCore newCandidate(scRef);
    newCandidate.setParentSuperCluster(scRef);
    if (validConversions_) {
      if (risolveAmbiguity_) {
        reco::ConversionRef bestRef = solveAmbiguity(conversionHandle, scRef);
        if (bestRef.isNonnull())
          newCandidate.addConversion(bestRef);

      } else {
        for (unsigned int icp = 0; icp < conversionHandle->size(); icp++) {
          reco::ConversionRef cpRef(reco::ConversionRef(conversionHandle, icp));
          if (cpRef->caloCluster().empty())
            continue;
          if (!(scRef.id() == cpRef->caloCluster()[0].id() && scRef.key() == cpRef->caloCluster()[0].key()))
            continue;
          if (!cpRef->isConverted())
            continue;
          newCandidate.addConversion(cpRef);
        }

      }  // solve or not the ambiguity of many conversion candidates
    }

    if (validPixelSeeds_) {
      for (unsigned int icp = 0; icp < pixelSeedHandle->size(); icp++) {
        reco::ElectronSeedRef cpRef(reco::ElectronSeedRef(pixelSeedHandle, icp));
        if (!cpRef->isEcalDriven())
          continue;
        if (!(scRef.id() == cpRef->caloCluster().id() && scRef.key() == cpRef->caloCluster().key()))
          continue;
        newCandidate.addElectronPixelSeed(cpRef);
      }
    }

    outputPhotonCoreCollection.push_back(newCandidate);
  }
}

reco::ConversionRef PhotonCoreProducer::solveAmbiguity(const edm::Handle<reco::ConversionCollection>& conversionHandle,
                                                       reco::SuperClusterRef& scRef) {
  std::multimap<reco::ConversionRef, double> convMap;
  for (unsigned int icp = 0; icp < conversionHandle->size(); icp++) {
    reco::ConversionRef cpRef(reco::ConversionRef(conversionHandle, icp));

    if (!(scRef.id() == cpRef->caloCluster()[0].id() && scRef.key() == cpRef->caloCluster()[0].key()))
      continue;
    if (!cpRef->isConverted())
      continue;
    double like = cpRef->MVAout();
    convMap.insert(std::make_pair(cpRef, like));
  }

  std::multimap<reco::ConversionRef, double>::iterator iMap;
  double max_lh = -1.;
  reco::ConversionRef bestRef;
  //  std::cout << " Pick up the best conv " << std::endl;
  for (iMap = convMap.begin(); iMap != convMap.end(); iMap++) {
    double like = iMap->second;
    if (like > max_lh) {
      max_lh = like;
      bestRef = iMap->first;
    }
  }

  //std::cout << " Best conv like " << max_lh << std::endl;

  float ep = 0;
  if (max_lh < 0) {
    //    std::cout << " Candidates with only one track " << std::endl;
    /// only one track reconstructed. Pick the one with best E/P
    float epMin = 999;

    for (iMap = convMap.begin(); iMap != convMap.end(); iMap++) {
      reco::ConversionRef convRef = iMap->first;
      // std::vector<reco::TrackRef> tracks = convRef->tracks();
      const std::vector<edm::RefToBase<reco::Track> > tracks = convRef->tracks();
      float px = tracks[0]->innerMomentum().x();
      float py = tracks[0]->innerMomentum().y();
      float pz = tracks[0]->innerMomentum().z();
      float p = sqrt(px * px + py * py + pz * pz);
      ep = fabs(1. - convRef->caloCluster()[0]->energy() / p);
      //    std::cout << " 1-E/P = " << ep << std::endl;
      if (ep < epMin) {
        epMin = ep;
        bestRef = iMap->first;
      }
    }
    //  std::cout << " Best conv 1-E/P " << ep << std::endl;
  }

  return bestRef;
}
