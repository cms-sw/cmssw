// -*- C++ -*-
//
// Package:    MonitorTrackGlobal
// Class:      MonitorTrackGlobal
// 
/**\class MonitorTrackGlobal MonitorTrackGlobal.cc DQM/TrackerMonitorTrack/src/MonitorTrackGlobal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Israel Goitom
//         Created:  Tue May 23 18:35:30 CEST 2006
// $Id: MonitorTrackGlobal.cc,v 1.12 2006/10/27 01:35:22 wmtan Exp $
//
//

#include <string>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackGlobal.h"

MonitorTrackGlobal::MonitorTrackGlobal(const edm::ParameterSet& iConfig)
{
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
  MTCCData = conf_.getParameter<bool>("MTCCData"); // if MTCC data certain histograms are not relevant
}

MonitorTrackGlobal::~MonitorTrackGlobal()
{
}

void MonitorTrackGlobal::beginJob(edm::EventSetup const& iSetup)
{
  using namespace edm;
  dbe->setCurrentFolder("Track/GlobalParameters");

  //    
  int TKNoBin = conf_.getParameter<int>("TkSizeBin");
  double TKNoMin = conf_.getParameter<double>("TkSizeMin");
  double TKNoMax = conf_.getParameter<double>("TkSizeMax");
  NumberOfTracks = dbe->book1D("NumberOfTracks", "NumberOfTracks.", TKNoBin, TKNoMin, TKNoMax);


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

  NumberOfRecHitsPerTrack = dbe->book1D("NumberOfRecHitsPerTrack", "NumberOfRecHitsPerTrack", TKHitBin, TKHitMin, TKHitMax);
  NumberOfMeanRecHitsPerTrack = dbe->book1D("NumberOfMeanRecHitsPerTrack", "NumberOfMeanRecHitsPerTrack", TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsPhi = dbe->book2D("NumberOfRecHitsPerTrackVsPhi","NumberOfRecHitsPerTrackVsPhi", PhiBin, PhiMin, PhiMax, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsTheta = dbe->book2D("NumberOfRecHitsPerTrackVsTheta","NumberOfRecHitsPerTrackVsTheta", ThetaBin, ThetaMin, ThetaMax, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrackVsEta = dbe->book2D("NumberOfRecHitsPerTrackVsEta","NumberOfRecHitsPerTrackVsEta", EtaBin, EtaMin, EtaMax, TKHitBin, TKHitMin, TKHitMax);

  //
  int Chi2Bin = conf_.getParameter<int>("Chi2Bin");
  double Chi2Min = conf_.getParameter<double>("Chi2Min");
  double Chi2Max = conf_.getParameter<double>("Chi2Max");

  Chi2 = dbe->book1D("Chi2", "Chi2", Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoF = dbe->book1D("Chi2overDoF", "Chi2overDoF", Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsTheta = dbe->book2D("Chi2overDoFVsTheta", "Chi2overDoFVsTheta", ThetaBin, ThetaMin, ThetaMax, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsPhi   = dbe->book2D("Chi2overDoFVsPhi"  , "Chi2overDoFVsPhi", PhiBin, PhiMin, PhiMax, Chi2Bin, Chi2Min, Chi2Max);
  Chi2overDoFVsEta   = dbe->book2D("Chi2overDoFVsEta"  , "Chi2overDoFVsEta", EtaBin, EtaMin, EtaMax, Chi2Bin, Chi2Min, Chi2Max);

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

  TrackPt = dbe->book1D("TrackPt", "TrackPt", TrackPtBin, TrackPtMin, TrackPtMax);
  TrackPx = dbe->book1D("TrackPx", "TrackPx", TrackPxBin, TrackPxMin, TrackPxMax);
  TrackPy = dbe->book1D("TrackPy", "TrackPy", TrackPyBin, TrackPyMin, TrackPyMax);
  TrackPz = dbe->book1D("TrackPz", "TrackPz", TrackPzBin, TrackPzMin, TrackPzMax);

  //dbe->setCurrentFolder("Tracker/Track Parameters");
  TrackPhi = dbe->book1D("TrackPhi", "TrackPhi.", PhiBin, PhiMin, PhiMax);
  TrackEta = dbe->book1D("TrackEta", "TrackEta.", EtaBin, EtaMin, EtaMax);
  TrackTheta = dbe->book1D("TrackTheta", "TrackTheta.", ThetaBin, ThetaMin, ThetaMax);

  if (!MTCCData)
    {
      DistanceOfClosestApproach = dbe->book1D("DistanceOfClosestApproach","DistanceOfClosestApproach",100, -0.5, 0.5);
      DistanceOfClosestApproachVsTheta = dbe->book2D("DistanceOfClosestApproachVsTheta","DistanceOfClosestApproachVsTheta", ThetaBin, ThetaMin, ThetaMax, 100, -0.4, 0.4);
      DistanceOfClosestApproachVsPhi = dbe->book2D("DistanceOfClosestApproachVsPhi","DistanceOfClosestApproachVsPhi", PhiBin, PhiMin, PhiMax, 100, -0.5, 0.5);
      DistanceOfClosestApproachVsEta = dbe->book2D("DistanceOfClosestApproachVsEta","DistanceOfClosestApproachVsEta", EtaBin, EtaMin, EtaMax, 100, -0.5, 0.5);

      xPointOfClosestApproach = dbe->book1D("xPointOfClosestApproach", "xPointOfClosestApproach", 20, -20, 20);
      yPointOfClosestApproach = dbe->book1D("yPointOfClosestApproach", "yPointOfClosestApproach", 20, -20, 20);
      zPointOfClosestApproach = dbe->book1D("zPointOfClosestApproach", "zPointOfClosestApproach", 50, -100, 100);
    }
}

// ------------ method called to produce the data  ------------
void
MonitorTrackGlobal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif

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


void MonitorTrackGlobal::endJob(void)
{
  dbe->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe->save(outputFileName);
  }
}

//define this as a plug-in
//DEFINE_FWK_MODULE(MonitorTrackGlobal);
