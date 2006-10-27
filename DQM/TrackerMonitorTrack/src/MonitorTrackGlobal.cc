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
// $Id: MonitorTrackGlobal.cc,v 1.11 2006/08/17 07:56:51 dkcira Exp $
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
  NumberOfTracks = dbe->book1D("NumberOfTracks", "NumberOfTracks.", 6, -0.5, 5.5);
  NumberOfRecHitsPerTrack = dbe->book1D("NumberOfRecHitsPerTrack", "NumberOfRecHitsPerTrack", 12, -0.5, 11.5);
  NumberOfRecHitsPerTrackVsPhi = dbe->book1D("NumberOfRecHitsPerTrackVsPhi","NumberOfRecHitsPerTrackVsPhi", 20, -3.142, 3.142);
  NumberOfRecHitsPerTrackVsPseudorapidity = dbe->book1D("NumberOfRecHitsPerTrackVsPseudorapidity","NumberOfRecHitsPerTrackVsPseudorapidity",20,-3.142,3.142);

  //
  Chi2 = dbe->book1D("Chi2", "Chi2", 20, 0, 100);
  Chi2overDoF = dbe->book1D("Chi2overDoF", "Chi2overDoF", 20, 0, 10);
  Chi2overDoFVsTheta = dbe->book2D("Chi2overDoFVsTheta", "Chi2overDoFVsTheta", 20, 0, 3.2 , 20, 0, 20);
  Chi2overDoFVsPhi   = dbe->book2D("Chi2overDoFVsPhi"  , "Chi2overDoFVsPhi", 20, -3.142 , 3.142 , 20, 0, 20);
  Chi2overDoFVsEta   = dbe->book2D("Chi2overDoFVsEta"  , "Chi2overDoFVsEta", 20, -3 , 3, 20, 0, 20);

  //dbe->setCurrentFolder("Tracker/Track Parameters");
  TrackPt = dbe->book1D("TrackPt", "TrackPt", 20, 0, 100);
  TrackPx = dbe->book1D("TrackPx", "TrackPx", 20, -100, 200);
  TrackPy = dbe->book1D("TrackPy", "TrackPy", 20, -100, 200);
  TrackPz = dbe->book1D("TrackPz", "TrackPz", 20, -100, 200);

  //dbe->setCurrentFolder("Tracker/Track Parameters");
  TrackPhi = dbe->book1D("TrackPhi", "TrackPhi.", 20, -3.142, 3.142);
  TrackEta = dbe->book1D("TrackEta", "TrackEta.", 20, -4, 4);
  TrackTheta = dbe->book1D("TrackTheta", "TrackTheta.", 20, -0.5, 4);

  if (!MTCCData)
    {
      DistanceOfClosestApproach = dbe->book1D("DistanceOfClosestApproach","DistanceOfClosestApproach",50,0,.2);
      DistanceOfClosestApproachVsTheta = dbe->book2D("DistanceOfClosestApproachVsTheta","DistanceOfClosestApproachVsTheta", 50, 0, 3.2, 50, 0, .2);
      DistanceOfClosestApproachVsPhi = dbe->book2D("DistanceOfClosestApproachVsPhi","DistanceOfClosestApproachVsPhi", 50, -3.142 , 3.142 , 50, 0, .2);
      DistanceOfClosestApproachVsEta = dbe->book2D("DistanceOfClosestApproachVsEta","DistanceOfClosestApproachVsEta", 50, -3 , 3 , 50, 0, .2);

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
   NumberOfRecHitsPerTrack->Fill(meanrechits);
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
