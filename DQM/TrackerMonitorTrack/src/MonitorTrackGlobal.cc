#include <string>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackGlobal.h"

MonitorTrackGlobal::MonitorTrackGlobal(const edm::ParameterSet& iConfig) {
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
  MTCCData = conf_.getParameter<bool>("MTCCData"); // if MTCC data certain histograms are not relevant
}

MonitorTrackGlobal::~MonitorTrackGlobal() { }

void MonitorTrackGlobal::beginJob(edm::EventSetup const& iSetup) {
  using namespace edm;

  std::string AlgoName = conf_.getParameter<std::string>("AlgoName");

  dbe->setCurrentFolder("Track/GlobalParameters");

  //    
  int TKNoBin = conf_.getParameter<int>("TkSizeBin");
  double TKNoMin = conf_.getParameter<double>("TkSizeMin");
  double TKNoMax = conf_.getParameter<double>("TkSizeMax");
  histname = "NumberOfTracks_";
  NumberOfTracks = dbe->book1D(histname+AlgoName, "NumberOfTracks.", TKNoBin, TKNoMin, TKNoMax);


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
  NumberOfRecHitsPerTrack = dbe->book1D(histname+AlgoName, "NumberOfRecHitsPerTrack", TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrack->setAxisTitle("Number of RecHits of each track");
  histname = "NumberOfMeanRecHitsPerTrack_";
  NumberOfMeanRecHitsPerTrack = dbe->book1D(histname+AlgoName, "NumberOfMeanRecHitsPerTrack", TKHitBin, TKHitMin, TKHitMax);
  NumberOfMeanRecHitsPerTrack->setAxisTitle("Mean number of RecHits per track");
  histname = "NumberOfRecHitsPerTrackVsPhi_";
  NumberOfRecHitsPerTrackVsPhi = dbe->book2D(histname+AlgoName,"NumberOfRecHitsPerTrackVsPhi", PhiBin, PhiMin, PhiMax, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track azimuthal angle",1);
  NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of RecHits of each track",2);
  histname = "NumberOfRecHitsPerTrackVsTheta_";
  NumberOfRecHitsPerTrackVsTheta = dbe->book2D(histname+AlgoName, "NumberOfRecHitsPerTrackVsTheta", ThetaBin, ThetaMin, ThetaMax, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Track polar angle",1);
  NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Number of RecHits of each track",2);
  histname = "NumberOfRecHitsPerTrackVsEta_";
  NumberOfRecHitsPerTrackVsEta = dbe->book2D(histname+AlgoName, "NumberOfRecHitsPerTrackVsEta", EtaBin, EtaMin, EtaMax, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track pseudorapidity",1);
  NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of RecHits of each track",2);

  //
  int Chi2Bin = conf_.getParameter<int>("Chi2Bin");
  double Chi2Min = conf_.getParameter<double>("Chi2Min");
  double Chi2Max = conf_.getParameter<double>("Chi2Max");

  histname = "Chi2_";
  Chi2 = dbe->book1D(histname+AlgoName, "Chi2", Chi2Bin, Chi2Min, Chi2Max);
  Chi2->setAxisTitle("Chi2 of each track");
  histname = "Chi2overDoF_";
  Chi2overDoF = dbe->book1D(histname+AlgoName, "Chi2overDoF", Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoF->setAxisTitle("Chi2 over nr. of degrees of freedom of each track");
  histname = "Chi2overDoFVsTheta_";
  Chi2overDoFVsTheta = dbe->book2D(histname+AlgoName, "Chi2overDoFVsTheta", ThetaBin, ThetaMin, ThetaMax, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsTheta->setAxisTitle("Track polar angle",1);
  Chi2overDoFVsTheta->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);
  histname = "Chi2overDoFVsPhi_";
  Chi2overDoFVsPhi   = dbe->book2D(histname+AlgoName, "Chi2overDoFVsPhi", PhiBin, PhiMin, PhiMax, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsPhi->setAxisTitle("Track azimuthal angle",1);
  Chi2overDoFVsPhi->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);
  histname = "Chi2overDoFVsEta_";
  Chi2overDoFVsEta   = dbe->book2D(histname+AlgoName, "Chi2overDoFVsEta", EtaBin, EtaMin, EtaMax, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsEta->setAxisTitle("Track pseudorapidity",1);
  Chi2overDoFVsEta->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);

  //dbe->setCurrentFolder("Tracker/Track Parameters");
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
  TrackPt = dbe->book1D(histname+AlgoName, "TrackPt", TrackPtBin, TrackPtMin, TrackPtMax);
  TrackPt->setAxisTitle("Transverse track momentum");
  histname = "TrackPx_";
  TrackPx = dbe->book1D(histname+AlgoName, "TrackPx", TrackPxBin, TrackPxMin, TrackPxMax);
  TrackPx->setAxisTitle("x component of track momentum");
  histname = "TrackPy_";
  TrackPy = dbe->book1D(histname+AlgoName, "TrackPy", TrackPyBin, TrackPyMin, TrackPyMax);
  TrackPy->setAxisTitle("y component of track momentum");
  histname = "TrackPz_";
  TrackPz = dbe->book1D(histname+AlgoName, "TrackPz", TrackPzBin, TrackPzMin, TrackPzMax);
  TrackPz->setAxisTitle("z component of track momentum");

  //dbe->setCurrentFolder("Tracker/Track Parameters");
  histname = "TrackPhi_";
  TrackPhi = dbe->book1D(histname+AlgoName, "TrackPhi.", PhiBin, PhiMin, PhiMax);
  TrackPhi->setAxisTitle("Track azimuthal angle");
  histname = "TrackEta_";
  TrackEta = dbe->book1D(histname+AlgoName, "TrackEta.", EtaBin, EtaMin, EtaMax);
  TrackEta->setAxisTitle("Track pseudorapidity");
  histname = "TrackTheta_";
  TrackTheta = dbe->book1D(histname+AlgoName, "TrackTheta.", ThetaBin, ThetaMin, ThetaMax);
  TrackTheta->setAxisTitle("Track polar angle");

  if (!MTCCData) { // followint histograms not meaningful for MTCC data
      histname = "DistanceOfClosestApproach_";
      DistanceOfClosestApproach = dbe->book1D(histname+AlgoName,"DistanceOfClosestApproach",100, -0.5, 0.5);
      DistanceOfClosestApproach->setAxisTitle("Track distance of closest approach");
      histname = "DistanceOfClosestApproachVsTheta_";
      DistanceOfClosestApproachVsTheta = dbe->book2D(histname+AlgoName,"DistanceOfClosestApproachVsTheta", ThetaBin, ThetaMin, ThetaMax, 100, -0.4, 0.4);
      DistanceOfClosestApproachVsTheta->setAxisTitle("Track polar angle",1);
      DistanceOfClosestApproachVsTheta->setAxisTitle("Track distance of closest approach",2);
      histname = "DistanceOfClosestApproachVsPhi_";
      DistanceOfClosestApproachVsPhi = dbe->book2D(histname+AlgoName,"DistanceOfClosestApproachVsPhi", PhiBin, PhiMin, PhiMax, 100, -0.5, 0.5);
      DistanceOfClosestApproachVsPhi->setAxisTitle("Track azimuthal angle",1);
      DistanceOfClosestApproachVsPhi->setAxisTitle("Track distance of closest approach",2);
      histname = "DistanceOfClosestApproachVsEta_";
      DistanceOfClosestApproachVsEta = dbe->book2D(histname+AlgoName,"DistanceOfClosestApproachVsEta", EtaBin, EtaMin, EtaMax, 100, -0.5, 0.5);
      DistanceOfClosestApproachVsEta->setAxisTitle("Track pseudorapidity",1);
      DistanceOfClosestApproachVsEta->setAxisTitle("Track distance of closest approach",2);

      histname = "xPointOfClosestApproach_";
      xPointOfClosestApproach = dbe->book1D(histname+AlgoName, "xPointOfClosestApproach", 20, -20, 20);
      xPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the x-axis");
      histname = "yPointOfClosestApproach_";
      yPointOfClosestApproach = dbe->book1D(histname+AlgoName, "yPointOfClosestApproach", 20, -20, 20);
      yPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the y-axis");
      histname = "zPointOfClosestApproach_";
      zPointOfClosestApproach = dbe->book1D(histname+AlgoName, "zPointOfClosestApproach", 50, -100, 100);
      zPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the z-axis");
    }
}

void MonitorTrackGlobal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
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


void MonitorTrackGlobal::endJob(void) {
  dbe->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe->save(outputFileName);
  }
}
