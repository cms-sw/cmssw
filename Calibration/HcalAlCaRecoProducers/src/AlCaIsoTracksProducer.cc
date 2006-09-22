#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaIsoTracksProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TFile.h>

AlCaIsoTracksProducer::AlCaIsoTracksProducer(const edm::ParameterSet& iConfig)
{ 
  m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks");
  m_histoFlag = iConfig.getUntrackedParameter<int>("histoFlag",0);

  if(m_histoFlag==1){
    m_Hfile=new TFile("IsoHists.root","RECREATE");
    IsoHists.Ntrk = new TH1F("Ntrk","Number of tracks",51,-0.5,50.5);
    IsoHists.vx = new TH1F("Vertexx","Track vertex x",100,-0.25,0.25);
    IsoHists.vy = new TH1F("Vertexy","Track vertex y",100,-0.25,0.25);
    IsoHists.vz = new TH1F("Vertexz","Track vertex z",100,-20.,20.);
    IsoHists.eta = new TH1F("Eta","Track eta",100,-5.,5.);
    IsoHists.phi = new TH1F("Phi","Track phi",100,-3.5,3.5);
    IsoHists.p = new TH1F("Momentum","Track momentum",100,0.,20.);
    IsoHists.pt = new TH1F("pt","Track pt",100,0.,10.);
    IsoHists.Dvertx = new TH1F("Dvertx","Distance in vertex x",100,0.,0.2);
    IsoHists.Dverty = new TH1F("Dverty","Distance in vertex y",100,0.,0.2);
    IsoHists.Dvertz = new TH1F("Dvertz","Distance in vertex z",100,0.,0.5);
    IsoHists.Dvert = new TH1F("Dvert","Distance in vertex",100,0.,0.5);
    IsoHists.Deta = new TH1F("Deta","Distance in eta",100,0.,5.);
    IsoHists.Dphi = new TH1F("Dphi","Distance in phi",100,0.,3.5);
    IsoHists.Ddir = new TH1F("Ddir","Distance in eta-phi",100,0.,7.);
    IsoHists.Nisotr = new TH1F("Nisotr","No of isolated tracks",51,-0.5,50.5);
  }
//register your products
  produces<reco::TrackCollection>("IsoTracks");
  produces<reco::TrackExtraCollection>("IsoTracksExtra");
}


AlCaIsoTracksProducer::~AlCaIsoTracksProducer()
{
/*  IsoHists.Ntrk->Delete();
  IsoHists.vx->Delete();
  IsoHists.vy->Delete();
  IsoHists.vz->Delete();
  IsoHists.eta->Delete();
  IsoHists.phi->Delete();
  IsoHists.p->Delete();
  IsoHists.pt->Delete();
  IsoHists.Dvertx->Delete();
  IsoHists.Dverty->Delete();
  IsoHists.Dvertz->Delete();
  IsoHists.Dvert->Delete();
  IsoHists.Deta->Delete();
  IsoHists.Dphi->Delete();
  IsoHists.Ddir->Delete();
  IsoHists.Nisotr->Delete(); */
}


void
AlCaIsoTracksProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
    
   edm::Handle<reco::TrackCollection> trackCollection;
   edm::Handle<reco::TrackExtraCollection> trackExtraCollection;

   iEvent.getByLabel(m_inputTrackLabel,trackCollection);
   iEvent.getByLabel(m_inputTrackLabel,trackExtraCollection);
//   try {
//     iEvent.getByType(trackCollection);
//   } catch ( std::exception& ex ) {
//     LogDebug("") << "AlCaIsoTracksProducer: Error! can't get product!" << std::endl;
//   }
   const reco::TrackCollection tC = *(trackCollection.product());
   const reco::TrackExtraCollection tXC = *(trackExtraCollection.product());

//Create empty output collections
   std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
   std::auto_ptr<reco::TrackExtraCollection> outputTXColl(new reco::TrackExtraCollection);

   int itrk=0;
   int nisotr=0;
   reco::TrackExtraCollection::const_iterator tkx=tXC.begin();
   for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++) {
        int isol = 1;
        int itrk1=0;
        itrk++;
        double px = track->px();
        double py = track->py();
        double pz = track->pz();
        double ptrack = sqrt(px*px+py*py+pz*pz);
        if(m_histoFlag==1){
          IsoHists.vx->Fill(track->x());
          IsoHists.vy->Fill(track->y());
          IsoHists.vz->Fill(track->z());
          IsoHists.eta->Fill(track->outerEta());
          IsoHists.phi->Fill(track->outerPhi());
          IsoHists.p->Fill(ptrack);
          IsoHists.pt->Fill(track->pt());
        }
//            cout<<"Checking track "<<itrk<<std::endl;
            for (reco::TrackCollection::const_iterator track1=tC.begin(); track1!=tC.end(); track1++)
            {
               itrk1++;
               if (track == track1) continue;
               double dx = fabs(track->x()-track1->x());
               double dy = fabs(track->y()-track1->y());
               double dz = fabs(track->z()-track1->z());
               double drvert = sqrt(dx*dx+dy*dy+dz*dz);
//               cout <<" ...with track "<<itrk1<<": drvert ="<<drvert;
               if(m_histoFlag==1){
                 IsoHists.Dvertx->Fill(dx);
                 IsoHists.Dverty->Fill(dy);
                 IsoHists.Dvertz->Fill(dz);
                 IsoHists.Dvert->Fill(drvert);
               }
               if(drvert > 0.1) {
//                 cout <<std::endl;
// I don't understand this, so I've commented it out
//                 continue;    
               }
               double deta = fabs(track->outerEta() - track1->outerEta());  
               double dphi = fabs(track->outerPhi() - track1->outerPhi());
               if (dphi > atan(1.)*4.) dphi = 8.*atan(1.) - dphi;
               double ddir = sqrt(deta*deta+dphi*dphi);
//               cout <<", ddir ="<<ddir<<std::endl;
               if(m_histoFlag==1){
                 IsoHists.Deta->Fill(deta);
                 IsoHists.Dphi->Fill(dphi);
                 IsoHists.Ddir->Fill(ddir);
               }
               if( ddir < 0.5 ) isol = 0;
            }
      if (track->outerEta() != tkx->outerEta() || track->outerPhi() != tkx->outerPhi()) {
        cout <<" *** Error: track and track extra do not match"<<std::endl;
        isol=0;
      }
      if (isol==1) {
        if(ptrack > 1.) {
//          cout <<"   ---> Track "<<itrk<<" is isolated!"<<std::endl;
          outputTColl->push_back(*track);
          outputTXColl->push_back(*tkx);
          nisotr++;
        }
      }
      tkx++;
   }
   if(m_histoFlag==1){
     IsoHists.Ntrk->Fill(itrk);
     IsoHists.Nisotr->Fill(nisotr);
   }

//Put selected information in the event
  iEvent.put( outputTColl, "IsoTracks");
  iEvent.put( outputTXColl, "IsoTracksExtra");
}

void AlCaIsoTracksProducer::endJob(void) {
  if(m_histoFlag==1){
    m_Hfile->cd();
    IsoHists.Ntrk->Write();
    IsoHists.vx->Write();
    IsoHists.vy->Write();
    IsoHists.vz->Write();
    IsoHists.eta->Write();
    IsoHists.phi->Write();
    IsoHists.p->Write();
    IsoHists.pt->Write();
    IsoHists.Dvertx->Write();
    IsoHists.Dverty->Write();
    IsoHists.Dvertz->Write();
    IsoHists.Dvert->Write();
    IsoHists.Deta->Write();
    IsoHists.Dphi->Write();
    IsoHists.Ddir->Write();
    IsoHists.Nisotr->Write();
    m_Hfile->Close();
  }
}
