#include <iostream>
#include <vector>
#include <memory>
/** \class GEDPhotonCoreProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonCoreProducer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include <Math/VectorUtil.h>
#include <vector>
#include "TLorentzVector.h"
#include "TMath.h"

GEDPhotonCoreProducer::GEDPhotonCoreProducer(const edm::ParameterSet& config) : 
  conf_(config)

{

  // use onfiguration file to setup input/output collection names
  pfEgammaCandidates_ = 
    consumes<reco::PFCandidateCollection>(conf_.getParameter<edm::InputTag>("pfEgammaCandidates"));
  pixelSeedProducer_ = 
    consumes<reco::ElectronSeedCollection>(conf_.getParameter<edm::InputTag>("pixelSeedProducer"));

  GEDPhotonCoreCollection_ = conf_.getParameter<std::string>("gedPhotonCoreCollection");



  // Register the product
  produces<reco::PhotonCoreCollection>(GEDPhotonCoreCollection_);

}

GEDPhotonCoreProducer::~GEDPhotonCoreProducer() {}




void GEDPhotonCoreProducer::produce(edm::Event &theEvent, const edm::EventSetup& theEventSetup) {


  using namespace edm;
  //  nEvt_++;

  reco::PhotonCoreCollection outputPhotonCoreCollection;
  std::auto_ptr< reco::PhotonCoreCollection > outputPhotonCoreCollection_p(new reco::PhotonCoreCollection);

  // Get the  PF refined cluster  collection
  Handle<reco::PFCandidateCollection> pfCandidateHandle;
  theEvent.getByToken(pfEgammaCandidates_,pfCandidateHandle);
  if (!pfCandidateHandle.isValid()) {
    edm::LogError("GEDPhotonCoreProducer") 
      << "Error! Can't get the pfEgammaCandidates";
  }


 // Get ElectronPixelSeeds
  validPixelSeeds_=true;
  Handle<reco::ElectronSeedCollection> pixelSeedHandle;
  reco::ElectronSeedCollection pixelSeeds;
  theEvent.getByToken(pixelSeedProducer_, pixelSeedHandle);
  if (!pixelSeedHandle.isValid()) {
    validPixelSeeds_=false;
  }


 
  //  std::cout <<  "  GEDPhotonCoreProducer::produce input PFcandidate size " <<   pfCandidateHandle->size() << std::endl;


  // Loop over PF candidates and get only photons
  reco::ElectronSeedCollection::const_iterator pixelSeedItr;
  for(unsigned int lCand=0; lCand < pfCandidateHandle->size(); lCand++) {
    reco::PFCandidateRef candRef (reco::PFCandidateRef(pfCandidateHandle,lCand));

    // Retrieve stuff from the pfPhoton
    reco::PFCandidateEGammaExtraRef pfPhoRef =  candRef->egammaExtraRef();
    reco::SuperClusterRef  refinedSC= pfPhoRef->superClusterRef();
    reco::SuperClusterRef  boxSC= pfPhoRef->superClusterPFECALRef();
    const reco::ConversionRefVector & doubleLegConv = pfPhoRef->conversionRef();
    const reco::ConversionRefVector & singleLegConv = pfPhoRef->singleLegConversionRef();
    reco::CaloClusterPtr refinedSCPtr= edm::refToPtr(refinedSC);

    //    std::cout << "newCandidate  doubleLegConv="<<doubleLegConv.size()<< std::endl;
    //std::cout << "newCandidate  singleLegConv="<<  pfPhoRef->singleLegConvTrackRef().size()<< std::endl;

    //////////
    reco::PhotonCore newCandidate;
    newCandidate.setPFlowPhoton(true);
    newCandidate.setStandardPhoton(false);
    newCandidate.setSuperCluster(refinedSC);
    newCandidate.setParentSuperCluster(boxSC);
    // fill conversion infos
    

    for(unsigned int lConv=0; lConv < doubleLegConv.size(); lConv++) {
      newCandidate.addConversion(doubleLegConv[lConv]);
    } 
    
    for(unsigned int lConv=0; lConv < singleLegConv.size(); lConv++) {
      newCandidate.addOneLegConversion(singleLegConv[lConv]);
    }     

    //    std::cout << "newCandidate pf refined SC energy="<< newCandidate.superCluster()->energy()<<std::endl;
    //std::cout << "newCandidate pf SC energy="<< newCandidate.parentSuperCluster()->energy()<<std::endl;
    //std::cout << "newCandidate  nconv2leg="<<newCandidate.conversions().size()<< std::endl;

    if ( validPixelSeeds_) {
      for( unsigned int icp = 0;  icp < pixelSeedHandle->size(); icp++) {
        reco::ElectronSeedRef cpRef(pixelSeedHandle,icp);
        if ( boxSC.isNonnull() && boxSC.id() == cpRef->caloCluster().id() && boxSC.key() == cpRef->caloCluster().key() ) {
          newCandidate.addElectronPixelSeed(cpRef);     
        }
      } 
    }

    outputPhotonCoreCollection.push_back(newCandidate);
  }

  // put the product in the event
  //  edm::LogInfo("GEDPhotonCoreProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCoreCollection_p->assign(outputPhotonCoreCollection.begin(),outputPhotonCoreCollection.end());
  theEvent.put( outputPhotonCoreCollection_p, GEDPhotonCoreCollection_);
  


  
} 
