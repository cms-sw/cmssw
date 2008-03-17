#include <string>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackingMonitor.h"

TrackingMonitor::TrackingMonitor(const edm::ParameterSet& iConfig) {
  dqmStore_ = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;
  MTCCData = conf_.getParameter<bool>("MTCCData"); // if MTCC data certain histograms are not relevant
}

TrackingMonitor::~TrackingMonitor() { }

void TrackingMonitor::beginJob(edm::EventSetup const& iSetup) {
  using namespace edm;

  std::string AlgoName = conf_.getParameter<std::string>("AlgoName");

  dqmStore_->setCurrentFolder("Track/GlobalParameters");

  //    
  int TKNoBin = conf_.getParameter<int>("TkSizeBin");
  double TKNoMin = conf_.getParameter<double>("TkSizeMin");
  double TKNoMax = conf_.getParameter<double>("TkSizeMax");
  histname = "NumberOfTracks_";
  NumberOfTracks = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKNoBin, TKNoMin, TKNoMax);


  int TKHitBin = conf_.getParameter<int>("RecHitBin");
  double TKHitMin = conf_.getParameter<double>("RecHitMin");
  double TKHitMax = conf_.getParameter<double>("RecHitMax");

  int PhiBin = conf_.getParameter<int>("PhiBin");
  double PhiMin = conf_.getParameter<double>("PhiMin");
  double PhiMax = conf_.getParameter<double>("PhiMax");

  int EtaBin = conf_.getParameter<int>("EtaBin");
  double EtaMin = conf_.getParameter<double>("EtaMin");
  double EtaMax = conf_.getParameter<double>("EtaMax");

  int ThetaBin = conf_.getParameter<int>("ThetaBin");
  double ThetaMin = conf_.getParameter<double>("ThetaMin");
  double ThetaMax = conf_.getParameter<double>("ThetaMax");

  histname = "NumberOfRecHitsPerTrack_";
  NumberOfRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrack->setAxisTitle("Number of RecHits of each track");
  histname = "NumberOfMeanRecHitsPerTrack_";
  NumberOfMeanRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfMeanRecHitsPerTrack->setAxisTitle("Mean number of RecHits per track");
  histname = "NumberOfRecHitsPerTrackVsPhi_";
  NumberOfRecHitsPerTrackVsPhi = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, PhiBin, PhiMin, PhiMax, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track azimuthal angle",1);
  NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of RecHits of each track",2);
  histname = "NumberOfRecHitsPerTrackVsTheta_";
  NumberOfRecHitsPerTrackVsTheta = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, ThetaBin, ThetaMin, ThetaMax, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Track polar angle",1);
  NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Number of RecHits of each track",2);
  histname = "NumberOfRecHitsPerTrackVsEta_";
  NumberOfRecHitsPerTrackVsEta = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, EtaBin, EtaMin, EtaMax, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track pseudorapidity",1);
  NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of RecHits of each track",2);

  //
  int Chi2Bin = conf_.getParameter<int>("Chi2Bin");
  double Chi2Min = conf_.getParameter<double>("Chi2Min");
  double Chi2Max = conf_.getParameter<double>("Chi2Max");

  histname = "Chi2_";
  Chi2 = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, Chi2Bin, Chi2Min, Chi2Max);
  Chi2->setAxisTitle("Chi2 of each track");
  histname = "Chi2overDoF_";
  Chi2overDoF = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoF->setAxisTitle("Chi2 over nr. of degrees of freedom of each track");
  histname = "Chi2overDoFVsTheta_";
  Chi2overDoFVsTheta = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, ThetaBin, ThetaMin, ThetaMax, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsTheta->setAxisTitle("Track polar angle",1);
  Chi2overDoFVsTheta->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);
  histname = "Chi2overDoFVsPhi_";
  Chi2overDoFVsPhi   = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, PhiBin, PhiMin, PhiMax, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsPhi->setAxisTitle("Track azimuthal angle",1);
  Chi2overDoFVsPhi->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);
  histname = "Chi2overDoFVsEta_";
  Chi2overDoFVsEta   = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, EtaBin, EtaMin, EtaMax, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsEta->setAxisTitle("Track pseudorapidity",1);
  Chi2overDoFVsEta->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);

  //dqmStore_->setCurrentFolder("Tracker/Track Parameters");
  int TrackPtBin = conf_.getParameter<int>("TrackPtBin");
  double TrackPtMin = conf_.getParameter<double>("TrackPtMin");
  double TrackPtMax = conf_.getParameter<double>("TrackPtMax");

  int TrackPxBin = conf_.getParameter<int>("TrackPxBin");
  double TrackPxMin = conf_.getParameter<double>("TrackPxMin");
  double TrackPxMax = conf_.getParameter<double>("TrackPxMax");

  int TrackPyBin = conf_.getParameter<int>("TrackPyBin");
  double TrackPyMin = conf_.getParameter<double>("TrackPyMin");
  double TrackPyMax = conf_.getParameter<double>("TrackPyMax");

  int TrackPzBin = conf_.getParameter<int>("TrackPzBin");
  double TrackPzMin = conf_.getParameter<double>("TrackPzMin");
  double TrackPzMax = conf_.getParameter<double>("TrackPzMax");

  histname = "TrackPt_";
  TrackPt = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TrackPtBin, TrackPtMin, TrackPtMax);
  TrackPt->setAxisTitle("Transverse track momentum");
  histname = "TrackPx_";
  TrackPx = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TrackPxBin, TrackPxMin, TrackPxMax);
  TrackPx->setAxisTitle("x component of track momentum");
  histname = "TrackPy_";
  TrackPy = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TrackPyBin, TrackPyMin, TrackPyMax);
  TrackPy->setAxisTitle("y component of track momentum");
  histname = "TrackPz_";
  TrackPz = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TrackPzBin, TrackPzMin, TrackPzMax);
  TrackPz->setAxisTitle("z component of track momentum");

  //dqmStore_->setCurrentFolder("Tracker/Track Parameters");
  histname = "TrackPhi_";
  TrackPhi = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, PhiBin, PhiMin, PhiMax);
  TrackPhi->setAxisTitle("Track azimuthal angle");
  histname = "TrackEta_";
  TrackEta = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, EtaBin, EtaMin, EtaMax);
  TrackEta->setAxisTitle("Track pseudorapidity");
  histname = "TrackTheta_";
  TrackTheta = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, ThetaBin, ThetaMin, ThetaMax);
  TrackTheta->setAxisTitle("Track polar angle");

  if (!MTCCData) { // followint histograms not meaningful for MTCC data
      histname = "DistanceOfClosestApproach_";
      DistanceOfClosestApproach = dqmStore_->book1D(histname+AlgoName,histname+AlgoName,100, -0.5, 0.5);
      DistanceOfClosestApproach->setAxisTitle("Track distance of closest approach");
      histname = "DistanceOfClosestApproachVsTheta_";
      DistanceOfClosestApproachVsTheta = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, ThetaBin, ThetaMin, ThetaMax, 100, -0.4, 0.4);
      DistanceOfClosestApproachVsTheta->setAxisTitle("Track polar angle",1);
      DistanceOfClosestApproachVsTheta->setAxisTitle("Track distance of closest approach",2);
      histname = "DistanceOfClosestApproachVsPhi_";
      DistanceOfClosestApproachVsPhi = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, PhiBin, PhiMin, PhiMax, 100, -0.5, 0.5);
      DistanceOfClosestApproachVsPhi->setAxisTitle("Track azimuthal angle",1);
      DistanceOfClosestApproachVsPhi->setAxisTitle("Track distance of closest approach",2);
      histname = "DistanceOfClosestApproachVsEta_";
      DistanceOfClosestApproachVsEta = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, EtaBin, EtaMin, EtaMax, 100, -0.5, 0.5);
      DistanceOfClosestApproachVsEta->setAxisTitle("Track pseudorapidity",1);
      DistanceOfClosestApproachVsEta->setAxisTitle("Track distance of closest approach",2);

      histname = "xPointOfClosestApproach_";
      xPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, 20, -20, 20);
      xPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the x-axis");
      histname = "yPointOfClosestApproach_";
      yPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, 20, -20, 20);
      yPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the y-axis");
      histname = "zPointOfClosestApproach_";
      zPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, 50, -100, 100);
      zPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the z-axis");
    }
}

void TrackingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
   using namespace edm;

   std::string TrackProducer = conf_.getParameter<std::string>("TrackProducer");
   std::string TrackLabel = conf_.getParameter<std::string>("TrackLabel");

   Handle<reco::TrackCollection> trackCollection;
   iEvent.getByLabel(TrackProducer, TrackLabel, trackCollection);
   NumberOfTracks->Fill(trackCollection->size());

   int totalRecHits = 0;
   for (reco::TrackCollection::const_iterator track = trackCollection->begin(); track!=trackCollection->end(); ++track)
   {
     TrackPx->Fill(track->px());
     TrackPy->Fill(track->py());
     TrackPz->Fill(track->pz());
     TrackPt->Fill(track->pt());

     TrackPhi->Fill(track->phi());
     TrackEta->Fill(track->eta());
     TrackTheta->Fill(track->theta());

     Chi2->Fill(track->chi2());
     Chi2overDoF->Fill(track->normalizedChi2());
     Chi2overDoFVsTheta->Fill(track->theta(), track->normalizedChi2());
     Chi2overDoFVsPhi->Fill(track->phi(), track->normalizedChi2());
     Chi2overDoFVsEta->Fill(track->eta(), track->normalizedChi2());

     totalRecHits += track->recHitsSize();

     if (!MTCCData) // not relevant for MTCC data, so do not fill in that case
     {
       NumberOfRecHitsPerTrack->Fill(track->recHitsSize());
       NumberOfRecHitsPerTrackVsPhi->Fill(track->phi(), track->recHitsSize());
       NumberOfRecHitsPerTrackVsTheta->Fill(track->theta(), track->recHitsSize());
       NumberOfRecHitsPerTrackVsEta->Fill(track->eta(), track->recHitsSize());
       DistanceOfClosestApproach->Fill(track->d0());
       DistanceOfClosestApproachVsTheta->Fill(track->theta(), track->d0());
       DistanceOfClosestApproachVsPhi->Fill(track->phi(), track->d0());
       DistanceOfClosestApproachVsEta->Fill(track->eta(), track->d0());

       xPointOfClosestApproach->Fill(track->vertex().x());
       yPointOfClosestApproach->Fill(track->vertex().y());
       zPointOfClosestApproach->Fill(track->vertex().z());
     }
   }

   double meanrechits = 0;
   // check that track size to avoid division by zero.
   if (trackCollection->size()) meanrechits = static_cast<double>(totalRecHits)/static_cast<double>(trackCollection->size());
   NumberOfMeanRecHitsPerTrack->Fill(meanrechits);
}


void TrackingMonitor::endJob(void) {
  dqmStore_->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dqmStore_->save(outputFileName);
  }
}
DEFINE_FWK_MODULE(TrackingMonitor);
