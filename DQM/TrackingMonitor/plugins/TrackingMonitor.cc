/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/09/14 16:20:08 $
 *  $Revision: 1.6 $
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include "DQM/TrackingMonitor/plugins/TrackingMonitor.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include <string>

TrackingMonitor::TrackingMonitor(const edm::ParameterSet& iConfig) {
  dqmStore_ = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;
  builderName = conf_.getParameter<std::string>("TTRHBuilder");
  // the track analyzer
  theTrackAnalyzer = new TrackAnalyzer(conf_);
  
}

TrackingMonitor::~TrackingMonitor() { 
  delete theTrackAnalyzer;
}

void TrackingMonitor::beginJob(edm::EventSetup const& iSetup) {

  using namespace edm;

  std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

  dqmStore_->setCurrentFolder(MEFolderName);

  int    TKNoBin = conf_.getParameter<int>("TkSizeBin");
  double TKNoMin = conf_.getParameter<double>("TkSizeMin");
  double TKNoMax = conf_.getParameter<double>("TkSizeMax");

  int    TKNoSeedBin = conf_.getParameter<int>("TkSeedSizeBin");
  double TKNoSeedMin = conf_.getParameter<double>("TkSeedSizeMin");
  double TKNoSeedMax = conf_.getParameter<double>("TkSeedSizeMax");

  int    TKHitBin = conf_.getParameter<int>("RecHitBin");
  double TKHitMin = conf_.getParameter<double>("RecHitMin");
  double TKHitMax = conf_.getParameter<double>("RecHitMax");

  int    TKLayBin = conf_.getParameter<int>("RecLayBin");
  double TKLayMin = conf_.getParameter<double>("RecLayMin");
  double TKLayMax = conf_.getParameter<double>("RecLayMax");

  int    EtaBin   = conf_.getParameter<int>("EtaBin");
  double EtaMin   = conf_.getParameter<double>("EtaMin");
  double EtaMax   = conf_.getParameter<double>("EtaMax");

  int    PhiBin   = conf_.getParameter<int>("PhiBin");
  double PhiMin   = conf_.getParameter<double>("PhiMin");
  double PhiMax   = conf_.getParameter<double>("PhiMax");


  int    ThetaBin   = conf_.getParameter<int>("ThetaBin");
  double ThetaMin   = conf_.getParameter<double>("ThetaMin");
  double ThetaMax   = conf_.getParameter<double>("ThetaMax");
  dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");
 
  histname = "NumberOfTracks_";
  NumberOfTracks = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKNoBin, TKNoMin, TKNoMax);

  if (conf_.getParameter<bool>("doSeedParameterHistos")) {
  dqmStore_->setCurrentFolder(MEFolderName+"/TrackBuilding");

  histname = "NumberOfSeeds_";
  NumberOfSeeds = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKNoSeedBin, TKNoSeedMin, TKNoSeedMax);

    histname = "SeedEta_";
    SeedEta = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, EtaBin, EtaMin, EtaMax);
    SeedEta->setAxisTitle("Seed pseudorapidity");
    
    histname = "SeedPhi_";
    SeedPhi = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, PhiBin, PhiMin, PhiMax);
    SeedPhi->setAxisTitle("Seed azimuthal angle");
    
    histname = "SeedTheta_";
    SeedTheta = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, ThetaBin, ThetaMin, ThetaMax);
    SeedTheta->setAxisTitle("Seed polar angle");
  }

  if (conf_.getParameter<bool>("doSeedParameterHistos")) {
  histname = "NumberOfTrackCandidates_";
  NumberOfTrackCandidates = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKNoBin, TKNoMin, TKNoMax);
}
  dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

  histname = "NumberOfMeanRecHitsPerTrack_";
  NumberOfMeanRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfMeanRecHitsPerTrack->setAxisTitle("Mean number of RecHits per track");

  histname = "NumberOfMeanLayersPerTrack_";
  NumberOfMeanLayersPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLayBin, TKLayMin, TKLayMax);
  NumberOfMeanLayersPerTrack->setAxisTitle("Mean number of Layers per track");

  theTrackAnalyzer->beginJob(iSetup, dqmStore_);
 
}

//
// -- Analyse
//
void TrackingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);  

  InputTag trackProducer = conf_.getParameter<edm::InputTag>("TrackProducer");
  InputTag seedProducer = conf_.getParameter<edm::InputTag>("SeedProducer");
  InputTag tcProducer = conf_.getParameter<edm::InputTag>("TCProducer");
  InputTag bsSrc = conf_.getParameter< edm::InputTag >("beamSpot");
  Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(trackProducer, trackCollection);
  if (!trackCollection.isValid()) return;

  Handle<edm::View<TrajectorySeed> > seedCollection;
  Handle<TrackCandidateCollection> theTCCollection;
  
  if (conf_.getParameter<bool>("doSeedParameterHistos")) {
 

  iEvent.getByLabel(seedProducer, seedCollection);
  if (!seedCollection.isValid()) return;  
  iEvent.getByLabel(tcProducer, theTCCollection ); 
  if (!theTCCollection.isValid()) return;
}  
  Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      
  
 
  NumberOfTracks->Fill(trackCollection->size());
  if (conf_.getParameter<bool>("doSeedParameterHistos")) {

  NumberOfSeeds->Fill(seedCollection->size());

  NumberOfTrackCandidates->Fill(theTCCollection->size());
}      
  TrajectoryStateTransform tsTransform;
  TSCBLBuilderNoMaterial tscblBuilder;

  int totalRecHits = 0, totalLayers = 0;
  for (reco::TrackCollection::const_iterator track = trackCollection->begin(); track!=trackCollection->end(); ++track) {
  
    totalRecHits += track->found();
    totalLayers += track->hitPattern().trackerLayersWithMeasurement();

    theTrackAnalyzer->analyze(iEvent, iSetup, *track);
  }

  double meanrechits = 0, meanlayers = 0;
  // check that track size to avoid division by zero.
  if (trackCollection->size()) {
    meanrechits = static_cast<double>(totalRecHits)/static_cast<double>(trackCollection->size());
    meanlayers = static_cast<double>(totalLayers)/static_cast<double>(trackCollection->size());
  }
  NumberOfMeanRecHitsPerTrack->Fill(meanrechits);
  NumberOfMeanLayersPerTrack->Fill(meanlayers);

  if (conf_.getParameter<bool>("doSeedParameterHistos")) {
    iSetup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
    for(TrajectorySeedCollection::size_type i=0; i<seedCollection->size(); ++i){
      edm::RefToBase<TrajectorySeed> seed(seedCollection, i);
      TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*(seed->recHits().second-1));
      TrajectoryStateOnSurface state = tsTransform.transientState( seed->startingState(), recHit->surface(), theMF.product());
      if (!state.isValid()) continue;
      TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs);
      if (!tsAtClosestApproachSeed.isValid()) continue;
      
      GlobalVector pSeed = tsAtClosestApproachSeed.trackStateAtPCA().momentum();
      
      double etaSeed = state.globalMomentum().eta();
      SeedEta->Fill(etaSeed);
      double phiSeed  = pSeed.phi();
      SeedPhi->Fill(phiSeed);
      double thetaSeed  = pSeed.theta();
      SeedTheta->Fill(thetaSeed);
      
    }
  }
}


void TrackingMonitor::endJob(void) {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dqmStore_->showDirStructure();
    dqmStore_->save(outputFileName);
  }

  
}

DEFINE_FWK_MODULE(TrackingMonitor);
