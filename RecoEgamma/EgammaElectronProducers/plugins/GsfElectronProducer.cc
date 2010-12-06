// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      GsfElectronProducer
//
/**\class GsfElectronProducer RecoEgamma/ElectronProducers/src/GsfElectronProducer.cc

 Description: EDProducer of GsfElectron objects

 Implementation:
     <Notes on implementation>
*/

#include "GsfElectronProducer.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


#include <iostream>

using namespace reco;

/* void GsfElectronProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
 {
  edm::ParameterSetDescription desc ;

  // input collections
  desc.add<edm::InputTag>("gsfElectronCores",edm::InputTag("gsfElectronCores")) ;
  desc.add<edm::InputTag>("hcalTowers",edm::InputTag("towerMaker")) ;
  desc.add<edm::InputTag>("reducedBarrelRecHitCollection",edm::InputTag("ecalRecHit","EcalRecHitsEB")) ;
  desc.add<edm::InputTag>("reducedEndcapRecHitCollection",edm::InputTag("ecalRecHit","EcalRecHitsEE")) ;
  desc.add<edm::InputTag>("pfMVA",edm::InputTag("pfElectronTranslator:pf")) ;
  desc.add<edm::InputTag>("seedsTag",edm::InputTag("ecalDrivenElectronSeeds")) ;
  desc.add<edm::InputTag>("beamSpot",edm::InputTag("offlineBeamSpot")) ;

  // backward compatibility mechanism for ctf tracks
  desc.add<bool>("ctfTracksCheck",true) ;
  desc.add<edm::InputTag>("ctfTracks",edm::InputTag("generalTracks")) ;

  // steering
  desc.add<bool>("applyPreselection",true) ;
  desc.add<bool>("applyEtaCorrection",false) ;
  desc.add<bool>("applyAmbResolution",true) ;
  desc.add<unsigned>("ambSortingStrategy",1) ;
  desc.add<unsigned>("ambClustersOverlapStrategy",1) ;
  desc.add<bool>("addPflowElectrons",true) ;

  // preselection parameters (ecal driven electrons)
  desc.add<double>("minSCEtBarrel",4.0) ;
  desc.add<double>("minSCEtEndcaps",4.0) ;
  desc.add<double>("minEOverPBarrel",0.0) ;
  desc.add<double>("maxEOverPBarrel",999999999.) ;
  desc.add<double>("minEOverPEndcaps",0.0) ;
  desc.add<double>("maxEOverPEndcaps",999999999.) ;
  desc.add<double>("maxDeltaEtaBarrel",0.02) ;
  desc.add<double>("maxDeltaEtaEndcaps",0.02) ;
  desc.add<double>("maxDeltaPhiBarrel",0.15) ;
  desc.add<double>("maxDeltaPhiEndcaps",0.15) ;
  desc.add<double>("hOverEConeSize",0.15) ;
  desc.add<double>("hOverEPtMin",0.) ;
  desc.add<double>("maxHOverEBarrel",0.15) ;
  desc.add<double>("maxHOverEEndcaps",0.15) ;
  desc.add<double>("maxHBarrel",0.0) ;
  desc.add<double>("maxHEndcaps",0.0) ;
  desc.add<double>("maxSigmaIetaIetaBarrel",999999999.) ;
  desc.add<double>("maxSigmaIetaIetaEndcaps",999999999.) ;
  desc.add<double>("maxFbremBarrel",999999999.) ;
  desc.add<double>("maxFbremEndcaps",999999999.) ;
  desc.add<bool>("isBarrel",false) ;
  desc.add<bool>("isEndcaps",false) ;
  desc.add<bool>("isFiducial",false) ;
  desc.add<bool>("seedFromTEC",true) ;
  desc.add<double>("maxTIP",999999999.) ;
  desc.add<double>("minMVA",-0.4) ;

  // preselection parameters (tracker driven only electrons)
  desc.add<double>("minSCEtBarrelPflow",0.0) ;
  desc.add<double>("minSCEtEndcapsPflow",0.0) ;
  desc.add<double>("minEOverPBarrelPflow",0.0) ;
  desc.add<double>("maxEOverPBarrelPflow",999999999.) ;
  desc.add<double>("minEOverPEndcapsPflow",0.0) ;
  desc.add<double>("maxEOverPEndcapsPflow",999999999.) ;
  desc.add<double>("maxDeltaEtaBarrelPflow",999999999.) ;
  desc.add<double>("maxDeltaEtaEndcapsPflow",999999999.) ;
  desc.add<double>("maxDeltaPhiBarrelPflow",999999999.) ;
  desc.add<double>("maxDeltaPhiEndcapsPflow",999999999.) ;
  desc.add<double>("hOverEConeSizePflow",0.15) ;
  desc.add<double>("hOverEPtMinPflow",0.) ;
  desc.add<double>("maxHOverEBarrelPflow",999999999.) ;
  desc.add<double>("maxHOverEEndcapsPflow",999999999.) ;
  desc.add<double>("maxHBarrelPflow",0.0) ;
  desc.add<double>("maxHEndcapsPflow",0.0) ;
  desc.add<double>("maxSigmaIetaIetaBarrelPflow",999999999.) ;
  desc.add<double>("maxSigmaIetaIetaEndcapsPflow",999999999.) ;
  desc.add<double>("maxFbremBarrelPflow",999999999.) ;
  desc.add<double>("maxFbremEndcapsPflow",999999999.) ;
  desc.add<bool>("isBarrelPflow",false) ;
  desc.add<bool>("isEndcapsPflow",false) ;
  desc.add<bool>("isFiducialPflow",false) ;
  desc.add<double>("maxTIPPflow",999999999.) ;
  desc.add<double>("minMVAPflow",-0.4) ;

  // Isolation algos configuration
  desc.add<double>("intRadiusBarrelTk",0.015) ;
  desc.add<double>("intRadiusEndcapTk",0.015) ;
  desc.add<double>("stripBarrelTk",0.015) ;
  desc.add<double>("stripEndcapTk",0.015) ;
  desc.add<double>("ptMinTk",0.7) ;
  desc.add<double>("maxVtxDistTk",0.2) ;
  desc.add<double>("maxDrbTk",999999999.) ;
  desc.add<double>("intRadiusHcal",0.15) ;
  desc.add<double>("etMinHcal",0.0) ;
  desc.add<double>("intRadiusEcalBarrel",3.0) ;
  desc.add<double>("intRadiusEcalEndcaps",3.0) ;
  desc.add<double>("jurassicWidth",1.5) ;
  desc.add<double>("etMinBarrel",0.0) ;
  desc.add<double>("eMinBarrel",0.08) ;
  desc.add<double>("etMinEndcaps",0.1) ;
  desc.add<double>("eMinEndcaps",0.0) ;
  desc.add<bool>("vetoClustered",false) ;
  desc.add<bool>("useNumCrystals",true) ;
  desc.add<int>("severityLevelCut",4) ;
  desc.add<double>("severityRecHitThreshold",5.0) ;
  desc.add<double>("spikeIdThreshold",0.95) ;
  desc.add<std::string>("spikeIdString","kSwissCrossBordersIncluded") ;
  desc.add<std::vector<int> >("recHitFlagsToBeExcluded") ;

  edm::ParameterSetDescription descNested ;
  descNested.add<std::string>("propagatorAlongTISE","PropagatorWithMaterial") ;
  descNested.add<std::string>("propagatorOppositeTISE","PropagatorWithMaterialOpposite") ;
  desc.add<edm::ParameterSetDescription>("TransientInitialStateEstimatorParameters",descNested) ;

  // Corrections
  desc.add<std::string>("superClusterErrorFunction","EcalClusterEnergyUncertainty") ;

  descriptions.add("produceGsfElectrons",desc) ;
 }
 */
GsfElectronProducer::GsfElectronProducer( const edm::ParameterSet& iConfig )
{
  //register your products
  produces<GsfElectronCollection>();

  //create algo
  algo_ = new
    GsfElectronAlgo(iConfig,
		    iConfig.getParameter<double>("minSCEtBarrel"),
		    iConfig.getParameter<double>("minSCEtEndcaps"),
		    iConfig.getParameter<double>("maxEOverPBarrel"),
		    iConfig.getParameter<double>("maxEOverPEndcaps"),
		    iConfig.getParameter<double>("minEOverPBarrel"),
		    iConfig.getParameter<double>("minEOverPEndcaps"),
		    iConfig.getParameter<double>("maxDeltaEtaBarrel"),
		    iConfig.getParameter<double>("maxDeltaEtaEndcaps"),
		    iConfig.getParameter<double>("maxDeltaPhiBarrel"),
		    iConfig.getParameter<double>("maxDeltaPhiEndcaps"),
		    iConfig.getParameter<double>("maxSigmaIetaIetaBarrel"),
		    iConfig.getParameter<double>("maxSigmaIetaIetaEndcaps"),
		    iConfig.getParameter<double>("maxFbremBarrel"),
		    iConfig.getParameter<double>("maxFbremEndcaps"),
		    iConfig.getParameter<bool>("isBarrel"),
		    iConfig.getParameter<bool>("isEndcaps"),
		    iConfig.getParameter<bool>("isFiducial"),
		    iConfig.getParameter<bool>("seedFromTEC"),
		    iConfig.getParameter<double>("minMVA"),
		    iConfig.getParameter<double>("maxTIP"),
		    iConfig.getParameter<double>("minSCEtBarrelPflow"),
		    iConfig.getParameter<double>("minSCEtEndcapsPflow"),
		    iConfig.getParameter<double>("maxEOverPBarrelPflow"),
		    iConfig.getParameter<double>("maxEOverPEndcapsPflow"),
		    iConfig.getParameter<double>("minEOverPBarrelPflow"),
		    iConfig.getParameter<double>("minEOverPEndcapsPflow"),
		    iConfig.getParameter<double>("maxDeltaEtaBarrelPflow"),
		    iConfig.getParameter<double>("maxDeltaEtaEndcapsPflow"),
		    iConfig.getParameter<double>("maxDeltaPhiBarrelPflow"),
		    iConfig.getParameter<double>("maxDeltaPhiEndcapsPflow"),
		    iConfig.getParameter<double>("maxSigmaIetaIetaBarrelPflow"),
		    iConfig.getParameter<double>("maxSigmaIetaIetaEndcapsPflow"),
		    iConfig.getParameter<double>("maxFbremBarrelPflow"),
		    iConfig.getParameter<double>("maxFbremEndcapsPflow"),
		    iConfig.getParameter<bool>("isBarrelPflow"),
		    iConfig.getParameter<bool>("isEndcapsPflow"),
		    iConfig.getParameter<bool>("isFiducialPflow"),
		    iConfig.getParameter<double>("minMVAPflow"),
		    iConfig.getParameter<double>("maxTIPPflow"),
		    iConfig.getParameter<bool>("applyPreselection"),
		    iConfig.getParameter<bool>("applyEtaCorrection"),
		    iConfig.getParameter<bool>("applyAmbResolution"),
		    iConfig.getParameter<unsigned>("ambSortingStrategy"),
		    iConfig.getParameter<unsigned>("ambClustersOverlapStrategy"),
		    iConfig.getParameter<bool>("addPflowElectrons"),
		    iConfig.getParameter<double>("intRadiusBarrelTk"),
		    iConfig.getParameter<double>("intRadiusEndcapTk"),
		    iConfig.getParameter<double>("stripBarrelTk"),
		    iConfig.getParameter<double>("stripEndcapTk"),
		    iConfig.getParameter<double>("ptMinTk"),
		    iConfig.getParameter<double>("maxVtxDistTk"),
		    iConfig.getParameter<double>("maxDrbTk"),
		    iConfig.getParameter<double>("intRadiusHcal"),
		    iConfig.getParameter<double>("etMinHcal"),
		    iConfig.getParameter<double>("intRadiusEcalBarrel"),
		    iConfig.getParameter<double>("intRadiusEcalEndcaps"),
		    iConfig.getParameter<double>("jurassicWidth"),
		    iConfig.getParameter<double>("etMinBarrel"),
		    iConfig.getParameter<double>("eMinBarrel"),
		    iConfig.getParameter<double>("etMinEndcaps"),
		    iConfig.getParameter<double>("eMinEndcaps"),
		    iConfig.getParameter<bool>("vetoClustered"),
		    iConfig.getParameter<bool>("useNumCrystals"),
        iConfig.getParameter<int>("severityLevelCut"),
        iConfig.getParameter<double>("severityRecHitThreshold"),
        iConfig.getParameter<double>("spikeIdThreshold"),
        iConfig.getParameter<std::string>("spikeIdString"),
        iConfig.getParameter<std::vector<int> >("recHitFlagsToBeExcluded")
		    );

}

GsfElectronProducer::~GsfElectronProducer()
{
  delete algo_;
}

// ------------ method called to produce the data  ------------
void GsfElectronProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{
  algo_->setupES(iSetup);

  // Create the output collections
  std::auto_ptr<GsfElectronCollection> pOutEle(new GsfElectronCollection);

  // invoke algorithm
  algo_->run(e,*pOutEle);

  // put result into the Event
  e.put(pOutEle);

}


