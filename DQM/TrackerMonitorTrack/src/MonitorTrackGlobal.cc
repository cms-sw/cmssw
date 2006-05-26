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
// $Id: MonitorTrackGlobal.cc,v 1.3 2006/05/26 10:12:49 dkcira Exp $
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

   Handle<reco::TrackCollection> trackCollection;
   iEvent.getByLabel("TrackProducer", trackCollection);
   trackSize->Fill(trackCollection->size());

//reco::Track singletrack=*(tracks->begin());

 for (reco::TrackCollection::const_iterator track = trackCollection->begin(); track!=trackCollection->end(); ++track)
   {
     //LogInfo("Demo")<<"Track "<<(int)(track-trackCollection->begin())<<" Chisq: "<< track->normalizedChi2()<<" "<<track->recHitsSize()<<" hits";

     d0VsTheta->Fill(track->theta(), track->d0());
     d0VsPhi->Fill(track->phi(), track->d0());
     d0VsEta->Fill(track->eta(), track->d0());

     z0VsTheta->Fill(track->theta(), track->d0());
     z0VsPhi->Fill(track->phi(), track->d0());
     z0VsEta->Fill(track->eta(), track->d0());

     chiSqrdVsTheta->Fill(track->theta(), track->normalizedChi2());
     chiSqrdVsPhi->Fill(track->phi(), track->normalizedChi2());
     chiSqrdVsEta->Fill(track->eta(), track->normalizedChi2());

   }
}

void MonitorTrackGlobal::beginJob(edm::EventSetup const& iSetup)
{
  using namespace edm;

  dbe->setCurrentFolder("SiStrip/Track Parameters");
  trackSize = dbe->book1D("TkSize", "Track size.", 50, 0, 100);

  dbe->setCurrentFolder("SiStrip/Track Parameters/MomentumParameters");

  dbe->setCurrentFolder("SiStrip/Track Parameters/Impact Parameters/d0");
  d0VsTheta = dbe->book2D("d0 vs. #theta", "Transverse Impact Parameter VS #theta.", 50, 0, 3.2, 50, 0, .2);
  d0VsPhi = dbe->book2D("d0 vs. #phi", "Transverse Impact Parameter VS #phi.", 50, -4 , 4 , 50, 0, .2);
  d0VsEta = dbe->book2D("d0 vs. #eta", "Transverse Impact Parameter VS #eta.", 50, -3 , 3 , 50, 0, .2);

  dbe->setCurrentFolder("SiStrip/Track Parameters/Impact Parameters/z0");
  z0VsTheta = dbe->book2D("z0 vs #theta", "Z Impact Parameter VS #theta.", 50, 0, 3.2, 50, -20, 20);
  z0VsPhi = dbe->book2D("z0 vs #phi", "Z Impact Parameter VS #phi.", 50, -4, 4, 50, -20, 20);
  z0VsEta = dbe->book2D("z0 vs #eta", "Z Impact Parameter VS #eta.", 50, -3, 3, 50, -20, 20);
  
  dbe->setCurrentFolder("SiStrip/Track Parameters/Trajectory Parameters");
  chiSqrdVsTheta = dbe->book2D("#chi^{2} vs #theta", "#chi^{2} vs #theta.", 50, 0, 3.2 , 50, 0, 20);
  chiSqrdVsPhi = dbe->book2D("#chi^{2} vs #phi", "#chi^{2} vs #phi.", 50, -4 , 4 , 50, 0, 20);
  chiSqrdVsEta = dbe->book2D("#chi^{2} vs #eta", "#chi^{2} vs #eta.", 50, -3 , 3, 50, 0, 20);
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
