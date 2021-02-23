/** \class GEDPhotonCoreProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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
  auto pfCandidateHandle = edm::makeValid(event.getHandle(pfEgammaCandidates_));

  // Get ElectronPixelSeeds
  bool validPixelSeeds = true;
  auto pixelSeedHandle = event.getHandle(pixelSeedProducer_);
  if (!pixelSeedHandle.isValid()) {
    validPixelSeeds = false;
  }

  // Loop over PF candidates and get only photons
  for (auto const& cand : *pfCandidateHandle) {
    // Retrieve stuff from the pfPhoton
    auto const& pfPho = *cand.egammaExtraRef();
    reco::SuperClusterRef refinedSC = pfPho.superClusterRef();
    reco::SuperClusterRef boxSC = pfPho.superClusterPFECALRef();

    // Construct new PhotonCore
    outputPhotonCoreCollection.emplace_back();
    auto& newCandidate = outputPhotonCoreCollection.back();

    newCandidate.setPFlowPhoton(true);
    newCandidate.setStandardPhoton(false);
    newCandidate.setSuperCluster(refinedSC);
    newCandidate.setParentSuperCluster(boxSC);

    // Fill conversion infos
    for (auto const& conv : pfPho.conversionRef()) {
      newCandidate.addConversion(conv);
    }
    for (auto const& conv : pfPho.singleLegConversionRef()) {
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
  }

  // put the product in the event
  event.emplace(putToken_, std::move(outputPhotonCoreCollection));
}
