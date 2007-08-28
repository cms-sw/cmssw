#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaIsoTracksProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h" 
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h" 
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <boost/regex.hpp> 

using namespace edm;
using namespace std;
using namespace reco;


#include <TFile.h>

AlCaIsoTracksProducer::AlCaIsoTracksProducer(const edm::ParameterSet& iConfig)
{ 
  
  m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks");
  hoLabel_ = iConfig.getParameter<edm::InputTag>("hoInput");
//
// Ecal Collection
//  
  ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
  hbheLabel_= iConfig.getParameter<edm::InputTag>("hbheInput");
  
  m_dvCut = iConfig.getUntrackedParameter<double>("vtxCut",0.05);
  m_ddirCut = iConfig.getUntrackedParameter<double>("coneCut",0.5);
  m_pCut = iConfig.getUntrackedParameter<double>("pCut",2.);
  m_ptCut = iConfig.getUntrackedParameter<double>("ptCut",1.5);
  m_ecalCut = iConfig.getUntrackedParameter<double>("ecalCut",8.);
  m_histoFlag = iConfig.getUntrackedParameter<int>("histoFlag",0);
  
//
// Parameters for track associator   ===========================
//  
  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_.useDefaultPropagator();
// ===============================================================
  if(m_histoFlag==1){
    m_Hfile=new TFile("IsoHists.root","RECREATE");
    IsoHists.Ntrk = new TH1F("Ntrk","Number of tracks",51,-0.5,50.5);
    IsoHists.vx = new TH1F("Vertexx","Track vertex x",100,-0.25,0.25);
    IsoHists.vy = new TH1F("Vertexy","Track vertex y",100,-0.25,0.25);
    IsoHists.vz = new TH1F("Vertexz","Track vertex z",100,-20.,20.);
    IsoHists.vr = new TH1F("VertexR","Track vertex R",100,0.,100.);
    IsoHists.eta = new TH1F("Eta","Track eta",100,-5.,5.);
    IsoHists.phi = new TH1F("Phi","Track phi",100,-3.5,3.5);
    IsoHists.p = new TH1F("Momentum","Track momentum",100,0.,20.);
    IsoHists.pt = new TH1F("pt","Track pt",100,0.,10.);
    IsoHists.Dvertx = new TH1F("Dvertx","Distance in vertex x",100,0.,0.2);
    IsoHists.Dverty = new TH1F("Dverty","Distance in vertex y",100,0.,0.2);
    IsoHists.Dvertz = new TH1F("Dvertz","Distance in vertex z",100,0.,0.5);
    IsoHists.Dvert = new TH1F("Dvert","Distance in vertex",100,0.,0.5);
    IsoHists.Dtheta = new TH1F("Dtheta","Distance in theta",100,0.,5.);
    IsoHists.Dphi = new TH1F("Dphi","Distance in phi",100,0.,3.5);
    IsoHists.Ddir = new TH1F("Ddir","Distance in eta-phi",100,0.,7.);
    IsoHists.Nisotr = new TH1F("Nisotr","No of isolated tracks",51,-0.5,50.5);
    IsoHists.Dering = new TH1F("DEring","ECAL ring energy",50,0.,25.);
    IsoHists.eecal = new TH1F("Eecal","ECAL RecHit energy",50,0.,25.);
    IsoHists.ehcal = new TH1F("Ehcal","HCAL RecHit energy",50,0.,25.);
  }
//register your products
  produces<reco::TrackCollection>("IsoTrackTracksCollection");
  produces<EcalRecHitCollection>("IsoTrackEcalRecHitCollection");
  produces<HBHERecHitCollection>("IsoTrackHBHERecHitCollection");
  produces<HORecHitCollection>("IsoTrackHORecHitCollection");
  
  allowMissingInputs_ = true;
}


AlCaIsoTracksProducer::~AlCaIsoTracksProducer() { }


void
AlCaIsoTracksProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
//   
//Create empty output collections
//
   std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
   std::auto_ptr<EcalRecHitCollection> outputEColl(new EcalRecHitCollection);
   std::auto_ptr<HBHERecHitCollection> outputHColl(new HBHERecHitCollection);
   std::auto_ptr<HORecHitCollection> outputHOColl(new HORecHitCollection);
   
// Temporarily collection   =================================================
   
  std::auto_ptr<EcalRecHitCollection> miniEcalRecHitCollection(new EcalRecHitCollection);
  std::auto_ptr<HBHERecHitCollection> miniHBHERecHitCollection(new HBHERecHitCollection);
    std::vector<edm::InputTag>::const_iterator i;
    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
    try {
      
      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);

       for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
                                                recHit != (*ec).end(); ++recHit)
       {
           miniEcalRecHitCollection->push_back(*recHit);
       }
    } catch (cms::Exception& e) { // can't find it!
    if (!allowMissingInputs_) { 
         cout<<" No ECAL input "<<endl;
         throw e;
	 }
    }
    }
    try {

      edm::Handle<HBHERecHitCollection> hbhe;
      iEvent.getByLabel(hbheLabel_,hbhe);
      const HBHERecHitCollection Hithbhe = *(hbhe.product());
  for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)
        {
         miniHBHERecHitCollection->push_back(*hbheItr);    
        }
  } catch (cms::Exception& e) { // can't find it!
    if (!allowMissingInputs_) {cout<<" No HCAL input "<<endl;throw e;}
  }
// =================================================================================

   
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   geo = pG.product();
   edm::Handle<reco::TrackCollection> trackCollection;
   try {
   iEvent.getByLabel(m_inputTrackLabel,trackCollection);
   } catch (cms::Exception& e) { // can't find it!
        if (!allowMissingInputs_) {std::cout<<" No Tracks in event "<<std::endl; throw e;}
   }
   

   const reco::TrackCollection tC = *(trackCollection.product());
   
//   cout<<" IsoTrackProducer starts with N tracks = "<< tC.size() << endl;
   
//   if( tC.size() == 0 )
//   {
//     iEvent.put( outputTColl, "IsoTrackTracksCollection");
//     iEvent.put( outputEColl, "IsoTrackEcalRecHitCollection");
//     iEvent.put( outputHColl, "IsoTrackHBHERecHitCollection");
//     iEvent.put( outputHOColl, "IsoTrackHORecHitCollection");
//   }


   int itrk=0;
   int nisotr=0;

//   Parameters for TrackDetAssociator ================================
// For Low momentum tracks need to look for larger cone for search ====
// ====================================================================

   parameters_.useEcal = true;
   parameters_.useHcal = true;
   parameters_.useCalo = false;
   parameters_.dREcal = 0.1;
   parameters_.dRHcal = 1.;

// main loop over input tracks
   for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++) {
        int isol = 1;
        int itrk1=0;
        itrk++;
        double px = track->px();
        double py = track->py();
        double pz = track->pz();
        double ptrack = sqrt(px*px+py*py+pz*pz);
        double rvert = sqrt(track->vx()*track->vx()+track->vy()*track->vy()+track->vz()*track->vz());
	
        if(m_histoFlag==1){
          IsoHists.vx->Fill(track->vx());
          IsoHists.vy->Fill(track->vy());
          IsoHists.vz->Fill(track->vz());
          IsoHists.vr->Fill(rvert);
          IsoHists.p->Fill(ptrack);
          IsoHists.pt->Fill(track->pt());
        }
	
        if (ptrack < m_pCut || track->pt() < m_ptCut || fabs(track->vx()) > m_dvCut || fabs(track->vx()) > m_dvCut) continue;

        TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *track), parameters_);
	
        double etaecal=info.trkGlobPosAtEcal.eta();
        double phiecal=info.trkGlobPosAtEcal.phi();
        double thetaecal=2.*atan(1.)-asin(2.*exp(etaecal)/(1.+exp(2.*etaecal)));
        if(etaecal<0)thetaecal=-thetaecal;
	double ddR = 0.1;
	
        double eecal=info.coneEnergy(ddR,TrackDetMatchInfo::EcalRecHits);

        if(m_histoFlag==1){
          IsoHists.eta->Fill(etaecal);
          IsoHists.phi->Fill(phiecal);
        }
// check charged isolation from all other tracks
//            cout<<"Checking track "<<itrk<<std::endl;
            for (reco::TrackCollection::const_iterator track1=tC.begin(); track1!=tC.end(); track1++)
//            for (reco::TrackCollection::const_iterator track1 = track+1; track1!=tC.end(); track1++)	    
            {
               itrk1++;
               if (track == track1) continue;
               double ptrack1 = sqrt(track1->px()*track1->px()+track1->py()*track1->py()+track1->pz()*track1->pz());
               double dx = fabs(track->vx()-track1->vx());
               double dy = fabs(track->vy()-track1->vy());
               double dz = fabs(track->vz()-track1->vz());
               double drvert = sqrt(dx*dx+dy*dy+dz*dz);
//               cout <<" ...with track "<<itrk1<<": drvert ="<<drvert;

               TrackDetMatchInfo info1 = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *track1), parameters_);

               double etaecal1=info1.trkGlobPosAtEcal.eta();
               double phiecal1=info1.trkGlobPosAtEcal.phi();
               double thetaecal1=2.*atan(1.)-asin(2.*exp(etaecal1)/(1.+exp(2.*etaecal1)));
               if(etaecal1<0)thetaecal1=-thetaecal1;
	       
               if(m_histoFlag==1){
                 IsoHists.Dvertx->Fill(dx);
                 IsoHists.Dverty->Fill(dy);
                 IsoHists.Dvertz->Fill(dz);
                 IsoHists.Dvert->Fill(drvert);
               }
	       
               double dtheta = fabs(thetaecal - thetaecal1);  
               double dphi = fabs(phiecal - phiecal1);
               if (dphi > atan(1.)*4.) dphi = 8.*atan(1.) - dphi;
               double ddir = sqrt(dtheta*dtheta+dphi*dphi);
//               cout <<", ddir ="<<ddir<<std::endl;

               if(m_histoFlag==1){
                 IsoHists.Dtheta->Fill(dtheta);
                 IsoHists.Dphi->Fill(dphi);
                 IsoHists.Ddir->Fill(ddir);
               }
	       
// increase required separation for low momentum tracks
               double factor=1.;
               double factor1=1.;
               if(ptrack<10.)factor+=(10.-ptrack)/20.;
               if(ptrack1<10.)factor1+=(10.-ptrack1)/20.;

               if( ddir < m_ddirCut*factor*factor1 ) {isol = 0; break;}
            }

//      TrackDetMatchInfo info2;
//      
// here check neutral isolation

      int iflag = 0;
      
      if (isol==1) {
//        parameters_.dREcal = 0.5;
//        info2 = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *track), parameters_);
//        double eecal2 = info2.ConeEnergy(ddR,EcalRecHits);


	ddR = 0.5;
        double eecal2 = info.coneEnergy(ddR,TrackDetMatchInfo::EcalRecHits);
	
        double dEring = eecal2-eecal;
	
        if(dEring < m_ecalCut) iflag=1;
        if(m_histoFlag==1){
          IsoHists.Dering->Fill(dEring);
        }

// we have a good isolated track, so write it out
        if(iflag == 1) {
          cout <<"   ---> Track "<<itrk<<" is isolated!"<<std::endl;
          outputTColl->push_back(*track);
	  
          for (std::vector<EcalRecHit>::const_iterator ehit=miniEcalRecHitCollection->begin(); ehit!=miniEcalRecHitCollection->end(); ehit++) {
            GlobalPoint posH = geo->getPosition((*ehit).detid());
            double phihit = posH.phi();
            double etahit = posH.eta();
           
	   
	   double dphi = fabs(info.trkGlobPosAtEcal.phi() - phihit); 
	   if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	   double deta = fabs( info.trkGlobPosAtEcal.eta() - etahit); 
           double dr = sqrt(dphi*dphi + deta*deta);
           if(dr<1.)  {
//            cout<<" ECAL energy "<<(*ehit).energy()<<
//	                        " eta "<<posH.eta()<<" phi "<<posH.phi()<<endl;
              if(m_histoFlag==1){
                 IsoHists.eecal->Fill(ehit->energy());
              }
            outputEColl->push_back(*ehit);
	    }   
         }
	 
          for (std::vector<HBHERecHit>::const_iterator hhit=miniHBHERecHitCollection->begin(); hhit!=miniHBHERecHitCollection->end(); hhit++) {
            GlobalPoint posH = geo->getPosition((*hhit).detid());
            double phihit = posH.phi();
            double etahit = posH.eta();
           
	   
	    double dphi = fabs(info.trkGlobPosAtHcal.phi() - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    double deta = fabs( info.trkGlobPosAtHcal.eta() - etahit); 
            double dr = sqrt(dphi*dphi + deta*deta);
	    
            if(dr<1.)  {
	    
//            cout<<" HCAL energy "<<(*hhit).energy()<<
//	                           " eta "<<posH.eta()<<" phi "<<posH.phi()<<endl;

            if(m_histoFlag==1){
              IsoHists.ehcal->Fill(hhit->energy());
            }
            outputHColl->push_back(*hhit);
	    }
          }
	  
          nisotr++;
        }
      }
   } // end of main track loop

   if(m_histoFlag==1){
     IsoHists.Ntrk->Fill(itrk);
     IsoHists.Nisotr->Fill(nisotr);
   }

    if(outputTColl->size() > 0)
    {
//   Take HO collection
     try {
      edm::Handle<HORecHitCollection> ho;
      iEvent.getByLabel(hoLabel_,ho);
      const HORecHitCollection Hitho = *(ho.product());
        for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
        {
             outputHOColl->push_back(*hoItr);
        }
     } catch (cms::Exception& e) { // can't find it!
        if (!allowMissingInputs_) {std::cout<<" No HO collection "<<std::endl; throw e;}
     }
    }
//   std::cout<<" Size of IsoTrk collections "<<outputHColl->size()<<" "<<outputEColl->size()<<
//   " "<<outputTColl->size()<<" "<<outputHOColl->size()<<std::endl;

//Put selected information in the event

  iEvent.put( outputTColl, "IsoTrackTracksCollection");
//  cout<<" Point 1 "<<endl;
  iEvent.put( outputEColl, "IsoTrackEcalRecHitCollection");
//  cout<<" Point 2 "<<endl;
  iEvent.put( outputHColl, "IsoTrackHBHERecHitCollection");
//  cout<<" Point 3 "<<endl;
  iEvent.put( outputHOColl, "IsoTrackHORecHitCollection");
//  cout<<" Point 4 "<<endl;

}

void AlCaIsoTracksProducer::endJob(void) {
  if(m_histoFlag==1){
    m_Hfile->cd();
    IsoHists.Ntrk->Write();
    IsoHists.vx->Write();
    IsoHists.vy->Write();
    IsoHists.vz->Write();
    IsoHists.vr->Write();
    IsoHists.eta->Write();
    IsoHists.phi->Write();
    IsoHists.p->Write();
    IsoHists.pt->Write();
    IsoHists.Dvertx->Write();
    IsoHists.Dverty->Write();
    IsoHists.Dvertz->Write();
    IsoHists.Dvert->Write();
    IsoHists.Dtheta->Write();
    IsoHists.Dphi->Write();
    IsoHists.Ddir->Write();
    IsoHists.Nisotr->Write();
    IsoHists.Dering->Write();
    IsoHists.eecal->Write();
    IsoHists.ehcal->Write();
    m_Hfile->Close();
  }
}

