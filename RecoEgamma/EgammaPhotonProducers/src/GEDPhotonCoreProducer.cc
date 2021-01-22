/** \class GEDPhotonCoreProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GEDPhotonCoreProducer : public edm::stream::EDProducer<> {
public:
  GEDPhotonCoreProducer(const edm::ParameterSet& ps);

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfEgammaCandidates_;
  const edm::EDGetTokenT<reco::ElectronSeedCollection> pixelSeedProducer_;
  const edm::EDPutTokenT<reco::PhotonCoreCollection> putToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEDPhotonCoreProducer);

GEDPhotonCoreProducer::GEDPhotonCoreProducer(const edm::ParameterSet& config)
    : pfEgammaCandidates_{consumes(config.getParameter<edm::InputTag>("pfEgammaCandidates"))},
      pixelSeedProducer_{consumes(config.getParameter<edm::InputTag>("pixelSeedProducer"))},
      putToken_{produces<reco::PhotonCoreCollection>(config.getParameter<std::string>("gedPhotonCoreCollection"))} {}

void GEDPhotonCoreProducer::produce(edm::Event& event, const edm::EventSetup&) {
  reco::PhotonCoreCollection outputPhotonCoreCollection;

  // Get the  PF refined cluster  collection
  auto pfCandidateHandle = event.getHandle(pfEgammaCandidates_);
  if (!pfCandidateHandle.isValid()) {
    edm::LogError("GEDPhotonCoreProducer") << "Error! Can't get the pfEgammaCandidates";
  }

  // Get ElectronPixelSeeds
  bool validPixelSeeds = true;
  auto pixelSeedHandle = event.getHandle(pixelSeedProducer_);
  if (!pixelSeedHandle.isValid()) {
    validPixelSeeds = false;
  }

  //  std::cout <<  "  GEDPhotonCoreProducer::produce input PFcandidate size " <<   pfCandidateHandle->size() << std::endl;

  // Loop over PF candidates and get only photons
  for (unsigned int lCand = 0; lCand < pfCandidateHandle->size(); lCand++) {
    reco::PFCandidateRef candRef(reco::PFCandidateRef(pfCandidateHandle, lCand));

    // Retrieve stuff from the pfPhoton
    reco::PFCandidateEGammaExtraRef pfPhoRef = candRef->egammaExtraRef();
    reco::SuperClusterRef refinedSC = pfPhoRef->superClusterRef();
    reco::SuperClusterRef boxSC = pfPhoRef->superClusterPFECALRef();

    //////////
    reco::PhotonCore newCandidate;
    newCandidate.setPFlowPhoton(true);
    newCandidate.setStandardPhoton(false);
    newCandidate.setSuperCluster(refinedSC);
    newCandidate.setParentSuperCluster(boxSC);

    // fill conversion infos
    for (auto const& conv : pfPhoRef->conversionRef()) {
      newCandidate.addConversion(conv);
    }
    for (auto const& conv : pfPhoRef->singleLegConversionRef()) {
      newCandidate.addOneLegConversion(conv);
    }

    //    std::cout << "newCandidate pf refined SC energy="<< newCandidate.superCluster()->energy()<<std::endl;
    //std::cout << "newCandidate pf SC energy="<< newCandidate.parentSuperCluster()->energy()<<std::endl;
    //std::cout << "newCandidate  nconv2leg="<<newCandidate.conversions().size()<< std::endl;

    if (validPixelSeeds) {
      for (unsigned int icp = 0; icp < pixelSeedHandle->size(); icp++) {
        reco::ElectronSeedRef cpRef(pixelSeedHandle, icp);
        if (boxSC.isNonnull() && boxSC.id() == cpRef->caloCluster().id() && boxSC.key() == cpRef->caloCluster().key()) {
          newCandidate.addElectronPixelSeed(cpRef);
        }
      }
    }

    outputPhotonCoreCollection.push_back(newCandidate);
  }

  // put the product in the event
  //  edm::LogInfo("GEDPhotonCoreProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  event.emplace(putToken_, std::move(outputPhotonCoreCollection));
}
