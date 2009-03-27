/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/25 01:30:15 $
 *  $Revision: 1.12 $
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include "DQM/TrackingMonitor/plugins/TrackingMonitor.h"

#include <string>

TrackingMonitor::TrackingMonitor(const edm::ParameterSet& iConfig) {
  dqmStore_ = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;

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

  int    TKHitBin = conf_.getParameter<int>("RecHitBin");
  double TKHitMin = conf_.getParameter<double>("RecHitMin");
  double TKHitMax = conf_.getParameter<double>("RecHitMax");

  int    TKLayBin = conf_.getParameter<int>("RecLayBin");
  double TKLayMin = conf_.getParameter<double>("RecLayMin");
  double TKLayMax = conf_.getParameter<double>("RecLayMax");

  histname = "NumberOfTracks_";
  NumberOfTracks = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKNoBin, TKNoMin, TKNoMax);

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
  
  InputTag trackProducer = conf_.getParameter<edm::InputTag>("TrackProducer");
  
  Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(trackProducer, trackCollection);
  if (!trackCollection.isValid()) return;

  NumberOfTracks->Fill(trackCollection->size());
  
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
