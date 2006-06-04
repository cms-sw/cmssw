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
// $Id: MonitorTrackGlobal.cc,v 1.5 2006/05/31 16:02:18 goitom Exp $
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
}


MonitorTrackGlobal::~MonitorTrackGlobal()
{
//  delete d0VsTheta;
//  delete d0VsPhi;
//  delete d0VsEta;
//  delete z0VsTheta;
//  delete z0VsPhi;
//  delete z0VsEta;
//  delete chiSqrdVsTheta;
//  delete chiSqrdVsPhi;
//  delete chiSqrdVsEta;
}


//
// member functions
//

void MonitorTrackGlobal::beginJob(edm::EventSetup const& iSetup)
{
  using namespace edm;

  dbe->setCurrentFolder("Tracker/Track Parameters");
  trackSize = dbe->book1D("Tracks Per Event", "Tracks Per Event.", 6, -1, 5);
  recHitSize = dbe->book1D("Mean RecHits Per Track", "Mean RecHits Per Track.", 15, -1, 35);
  chiSqrd = dbe->book1D("#chi^{2}", "#Chi^{2}", 50, 0, 100);

  //dbe->setCurrentFolder("Tracker/Track Parameters");
  trackPt = dbe->book1D("Track Transverse momentum", "Track Transverse momentum.", 20, 0, 1000);
  trackPX = dbe->book1D("Track X coordinate of momentum", "Track X coordinate of momentum.", 20, -800, 800);
  trackPY = dbe->book1D("Track Y coordinate of momentum", "Track Y coordinate of momentum.", 20, -800, 800);
  trackPZ = dbe->book1D("Track Z coordinate of momentum", "Track Z coordinate of momentum.", 20, 1000, 1000);

  //dbe->setCurrentFolder("Tracker/Track Parameters");
  trackPhi = dbe->book1D("Track Phi", "Track Phi.", 20, -4, 4);
  trackEta = dbe->book1D("Track Eta", "Track Eta.", 20, -4, 4);
  trackTheta = dbe->book1D("Track Theta", "Track Theta.", 20, -0.5, 4);

  bool MTCCData = conf_.getParameter<bool>("MTCCData");
  if (!MTCCData)
    {
      //dbe->setCurrentFolder("Tracker/Track Parameters");
      d0VsTheta = dbe->book2D("Distance of Closest Approach VS #theta", "Distance of Closest Approach VS #theta.", 50, 0, 3.2, 50, 0, .2);
      d0VsPhi = dbe->book2D("Distance of Closest Approach VS #phi", "Distance of Closest Approach VS #phi.", 50, -4 , 4 , 50, 0, .2);
      d0VsEta = dbe->book2D("Distance of Closest Approach VS #eta",  "Distance of Closest Approach VS #eta.", 50, -3 , 3 , 50, 0, .2);

      z0VsTheta = dbe->book2D("Z Coordinate Of Point Of Closest Approach VS #theta", "Z Coordinate Of Point Of Closest Approach VS #theta.", 50, 0, 3.2, 50, -20, 20);
      z0VsPhi = dbe->book2D("Z Coordinate Of Point Of Closest Approach VS #phi", "Z Coordinate Of Point Of Closest Approach VS #phi.", 50, -4, 4, 50, -20, 20);
      z0VsEta = dbe->book2D("Z Coordinate Of Point Of Closest Approach VS #eta", "Z Coordinate Of Point Of Closest Approach VS #eta.", 50, -3, 3, 50, -20, 20);
  
      //dbe->setCurrentFolder("Tracker/Track Parameters");
      chiSqrdVsTheta = dbe->book2D("#chi^{2} vs #theta", "#chi^{2} vs #theta.", 50, 0, 3.2 , 50, 0, 20);
      chiSqrdVsPhi = dbe->book2D("#chi^{2} vs #phi", "#chi^{2} vs #phi.", 50, -4 , 4 , 50, 0, 20);
      chiSqrdVsEta = dbe->book2D("#chi^{2} vs #eta", "#chi^{2} vs #eta.", 50, -3 , 3, 50, 0, 20);
    }

  //dbe->setCurrentFolder("Tracker/Track Parameters");
  trackVertexX = dbe->book1D("X Coordinate of Track vertex", "X Coordinate of Track vertex.", 20, -20, 20);
  trackVertexY = dbe->book1D("Y Coordinate of Track vertex", "Y Coordinate of Track vertex.", 20, -20, 20);
  trackVertexZ = dbe->book1D("Z Coordinate of Track vertex", "Z Coordinate of Track vertex.", 50, -100, 100);

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
   trackSize->Fill(trackCollection->size());

   int totalRecHits = 0;

//reco::Track singletrack=*(tracks->begin());

 for (reco::TrackCollection::const_iterator track = trackCollection->begin(); track!=trackCollection->end(); ++track)
   {
     //LogInfo("Demo")<<"Track "<<(int)(track-trackCollection->begin())<<" Chisq: "<< track->normalizedChi2()<<" "<<track->recHitsSize()<<" hits";

     chiSqrd->Fill(track->normalizedChi2());
     trackPX->Fill(track->px());
     trackPY->Fill(track->py());
     trackPZ->Fill(track->pz());
     trackPt->Fill(track->pt());
     trackVertexX->Fill(track->vertex().x());
     trackVertexY->Fill(track->vertex().y());
     trackVertexZ->Fill(track->vertex().z());
     trackPhi->Fill(track->phi());
     trackEta->Fill(track->eta());
     trackTheta->Fill(track->theta());

  bool MTCCData = conf_.getParameter<bool>("MTCCData");
  if (!MTCCData)
    {
     d0VsTheta->Fill(track->theta(), track->d0());
     d0VsPhi->Fill(track->phi(), track->d0());
     d0VsEta->Fill(track->eta(), track->d0());

     z0VsTheta->Fill(track->theta(), track->dz());
     z0VsPhi->Fill(track->phi(), track->dz());
     z0VsEta->Fill(track->eta(), track->dz());
    }

     chiSqrdVsTheta->Fill(track->theta(), track->normalizedChi2());
     chiSqrdVsPhi->Fill(track->phi(), track->normalizedChi2());
     chiSqrdVsEta->Fill(track->eta(), track->normalizedChi2());

     totalRecHits += track->recHitsSize();
   }

 double meanrechits = 0;
 if (trackCollection->size()) // check that track size to avoid division by zero.
   {
     meanrechits = static_cast<double>(totalRecHits)/static_cast<double>(trackCollection->size());
   }else
   {
     meanrechits = 0;
   }
 recHitSize->Fill(meanrechits);
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
//DEFINE_FWK_MODULE(MonitorTrackGlobal)
