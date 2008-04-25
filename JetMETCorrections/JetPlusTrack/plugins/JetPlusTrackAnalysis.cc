#include "JetMETCorrections/JetPlusTrack/plugins/JetPlusTrackAnalysis.h"

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Provenance/interface/Provenance.h"

using namespace std;
namespace cms
{

JetPlusTrackAnalysis::JetPlusTrackAnalysis(const edm::ParameterSet& iConfig)
{
   cout<<" Start JetPlusTrackAnalysis now"<<endl;
   mCone = iConfig.getParameter<double>("Cone");
   cout<<" Start JetPlusTrackAnalysis Point 0"<<endl;
   mInputJetsCaloTower = iConfig.getParameter<edm::InputTag>("src1");
   cout<<" Start JetPlusTrackAnalysis Point 1"<<endl;
   mInputJetsGen = iConfig.getParameter<edm::InputTag>("src2");	
   cout<<" Start JetPlusTrackAnalysis Point 2"<<endl;  
   mInputJetsCorrected = iConfig.getParameter<edm::InputTag>("src3");
   cout<<" Start JetPlusTrackAnalysis Point 3"<<endl;
   m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks");
   cout<<" Start JetPlusTrackAnalysis Point 4"<<endl;
   hbhelabel_ = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
   cout<<" Start JetPlusTrackAnalysis Point 5"<<endl;
   holabel_ = iConfig.getParameter<edm::InputTag>("HORecHitCollectionLabel");
   cout<<" Start JetPlusTrackAnalysis Point 6"<<endl;
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   cout<<" Start JetPlusTrackAnalysis Point 7"<<endl;
   fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
   cout<<" Start JetPlusTrackAnalysis Point 8"<<endl;
   allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",false);
   cout<<" JetPlusTrackAnalysis constructor "<<endl;			  
}

JetPlusTrackAnalysis::~JetPlusTrackAnalysis()
{
    cout<<" JetPlusTrack destructor "<<endl;
}

void JetPlusTrackAnalysis::beginJob( const edm::EventSetup& iSetup)
{

   cout<<" Begin job "<<endl;

   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
   myTree = new TTree("JetPlusTrack","JetPlusTrack Tree");
   myTree->Branch("run",  &run, "run/I");
   myTree->Branch("event",  &event, "event/I");

   NumRecoJetsCaloTower = 0;
   NumRecoJetsCorrected = 0;
   NumRecoJetsRecHit = 0;
   NumGenJets = 0;
   NumPart = 0;
   NumRecoTrack = 0;

// Jet Reco CaloTower
   myTree->Branch("NumRecoJetsCaloTower", &NumRecoJetsCaloTower, "NumRecoJetsCaloTower/I");
   myTree->Branch("JetRecoEtCaloTower",  JetRecoEtCaloTower, "JetRecoEtCaloTower[100]/F");
   myTree->Branch("JetRecoEtaCaloTower",  JetRecoEtaCaloTower, "JetRecoEtaCaloTower[100]/F");
   myTree->Branch("JetRecoPhiCaloTower",  JetRecoPhiCaloTower, "JetRecoPhiCaloTower[100]/F");
   myTree->Branch("JetRecoEtRecHit",  JetRecoEtRecHit, "JetRecoEtRecHit[100]/F");
   myTree->Branch("JetRecoGenRecType", JetRecoGenRecType, "JetRecoGenRecType[100]/F");
   myTree->Branch("JetRecoGenPartonType", JetRecoGenPartonType , "JetRecoGenPartonType[100]/F");
   myTree->Branch("EcalEmpty", EcalEmpty , "EcalEmpty[100]/F");
   myTree->Branch("HcalEmpty", HcalEmpty , "HcalEmpty[100]/F");
//
   myTree->Branch("NumRecoJetsCorrected", &NumRecoJetsCorrected, "NumRecoJetsCorrected/I");
   myTree->Branch("JetRecoEtCorrected",  JetRecoEtCorrected, "JetRecoEtCorrected[100]/F");
   myTree->Branch("JetRecoEtCorrectedZS",  JetRecoEtCorrectedZS, "JetRecoEtCorrectedZS[100]/F");
   myTree->Branch("JetRecoEtaCorrected",  JetRecoEtaCorrected, "JetRecoEtaCorrected[100]/F");
   myTree->Branch("JetRecoPhiCorrected",  JetRecoPhiCorrected, "JetRecoPhiCorrected[100]/F");
// GenJet block
   myTree->Branch("NumGenJets", &NumGenJets, "NumGenJets/I");
   myTree->Branch("JetGenEt",  JetGenEt, "JetGenEt[10]/F");
   myTree->Branch("JetGenEta",  JetGenEta, "JetGenEta[10]/F");
   myTree->Branch("JetGenPhi",  JetGenPhi, "JetGenPhi[10]/F");
   myTree->Branch("JetGenCode",  JetGenCode, "JetGenCode[10]/I");
// Particle block
   myTree->Branch("NumPart", &NumPart, "NumPart/I");
   myTree->Branch("Code",  Code, "Code[4000]/I");
   myTree->Branch("Charge",  Charge, "Charge[4000]/I");
   myTree->Branch("partpx",  partpx, "partpx[4000]/F");
   myTree->Branch("partpy",  partpy, "partpy[4000]/F");
   myTree->Branch("partpz",  partpz, "partpz[4000]/F");
   myTree->Branch("parte",  parte, "parte[4000]/F");
   myTree->Branch("partm",  partm, "partm[4000]/F");  
// Tracks block
   myTree->Branch("NumRecoTrack", &NumRecoTrack, "NumRecoTrack/I");
   myTree->Branch("TrackRecoEt",  TrackRecoEt, "TrackRecoEt[NumRecoTrack]/F");
   myTree->Branch("TrackRecoEta",  TrackRecoEta, "TrackRecoEta[NumRecoTrack]/F");
   myTree->Branch("TrackRecoPhi",  TrackRecoPhi, "TrackRecoPhi[NumRecoTrack]/F");

// Calo Geometry
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   geo = pG.product();
   

}
void JetPlusTrackAnalysis::endJob()
{

   cout << "===== Start writing user histograms =====" << endl;
   hOutputFile->SetCompressionLevel(2);
   hOutputFile->cd();
   myTree->Write();
   hOutputFile->Close() ;
   cout << "===== End writing user histograms =======" << endl;
   
}

void JetPlusTrackAnalysis::analyze(
                                         const edm::Event& iEvent,
                                         const edm::EventSetup& theEventSetup)  
{
    cout<<" JetPlusTrack analyze "<<endl;
//  std::vector<edm::Provenance const*> theProvenance;
//  iEvent.getAllProvenance(theProvenance);
//  for( std::vector<edm::Provenance const*>::const_iterator ip = theProvenance.begin();
//                                                      ip != theProvenance.end(); ip++)
//  {
//     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
//     " "<<(**ip).productInstanceName()<<endl;
//  }

    


   run = iEvent.id().run();
   event = iEvent.id().event();
//
//  Rememeber parton
//
  float pt[2],eta[2],phi[2];
  int parton[2];
  int tagparton;
  cout<<" Try to take HepMCProduct "<<endl;
  edm::Handle< edm::HepMCProduct >  EvtHandles ;
  iEvent.getByType( EvtHandles ) ;

//NR==================================================

  if (!EvtHandles.isValid()) {
    // can't find it!
    if (!allowMissingInputs_) {cout<<" GenParticles are missed "<<endl;}
    *EvtHandles;  // will throw the proper exception
  } else {
         const HepMC::GenEvent* Evt = EvtHandles->GetEvent() ;

        int ihep = 0; 
         
         for (HepMC::GenEvent::particle_const_iterator
            Part = Evt->particles_begin() ; Part!=Evt->particles_end(); Part++ )
         {
             if(ihep == 6 || ihep == 7)
             {
                cout<<" parton "<<(*Part)->pdg_id()<<" "<<(*Part)->status()<<" "<<((*Part)->momentum()).perp()<<endl;
                pt[ihep-6] = ((*Part)->momentum()).perp();
                eta[ihep-6] = ((*Part)->momentum()).eta();
                phi[ihep-6] = ((*Part)->momentum()).phi();
                parton[ihep-6] = (*Part)->pdg_id();
             } 
//             Code[ihep] = (*Part)->pdg_id();
//             partpx[ihep] = (*Part)->momentum().px();
//             partpy[ihep] = (*Part)->momentum().py();
//             partpz[ihep] = (*Part)->momentum().pz();
//             parte[ihep] = (*Part)->momentum().e();
//             partm[ihep] = (*Part)->momentum().m();
             ihep++;
//             NumPart = ihep;
         }
  }
//  Generated jet
   NumGenJets = 0;
   int icode = -1;
   {
   edm::Handle<reco::GenJetCollection> jets;
   iEvent.getByLabel(mInputJetsGen, jets);
   if (!jets.isValid()) {
     // can't find it!
     if (!allowMissingInputs_) {
       *jets;  // will throw the proper exception
     }
   } else {
     reco::GenJetCollection::const_iterator jet = jets->begin ();
     if(jets->size() > 0 )
       {
         for (; jet != jets->end (); jet++)
	   {
	     if( NumGenJets < 4 )
	       {
		 // Find the parton and associated jet
		 double dphi1 = fabs((*jet).phi()-phi[0]);
		 if(dphi1 > 4.*atan(1.)) dphi1 = 8.*atan(1.) - dphi1;
		 double dphi2 = fabs((*jet).phi()-phi[1]);
		 if(dphi2 > 4.*atan(1.)) dphi2 = 8.*atan(1.) - dphi2;
		 double deta1 = (*jet).eta()-eta[0];
		 double deta2 = (*jet).eta()-eta[1];
		 double dr1 = sqrt(dphi1*dphi1+deta1*deta1);
		 double dr2 = sqrt(dphi2*dphi2+deta2*deta2); 
		 if(dr1 < 0.5 || dr2 < 0.5) {
		   JetGenEt[NumGenJets] = (*jet).et();
		   JetGenEta[NumGenJets] = (*jet).eta();
		   JetGenPhi[NumGenJets] = (*jet).phi();
		   cout<<" Associated jet: Phi, eta gen "<< JetGenPhi[NumGenJets]<<" "<< JetGenEta[NumGenJets]<<endl;
		   
		   if(dr1 < 0.5) icode = 0;
		   if(dr2 < 0.5) icode = 1;
		   JetGenCode[NumGenJets] = icode;
		   cout<<" Gen jet "<<NumGenJets<<" "<<JetGenEt[NumGenJets]<<" "<<JetGenEta[NumGenJets]<<" "<<JetGenPhi[NumGenJets]<<" "<<JetGenCode[NumGenJets]<<endl;
		   NumGenJets++;
		 } 
	       }
	   }
       }
   }
   }

// CaloJets

    NumRecoJetsCaloTower = 0;
    {
    edm::Handle<reco::CaloJetCollection> jets;
    iEvent.getByLabel(mInputJetsCaloTower, jets);
    if (!jets.isValid()) {
      // can't find it!
      if (!allowMissingInputs_) {cout<<"CaloTowers are missed "<<endl; 
	*jets;  // will throw the proper exception
      }
    } else {
      reco::CaloJetCollection::const_iterator jet = jets->begin ();
      
      cout<<" Size of jets "<<jets->size()<<endl;
      
      if(jets->size() > 0 )
	{
	  for (; jet != jets->end (); jet++)
	    {
	      
	      if( NumRecoJetsCaloTower < 100 )
		{
		  
		  // Association with gen jet
		  
		  JetRecoEtCaloTower[NumRecoJetsCaloTower] = (*jet).et();
		  JetRecoEtaCaloTower[NumRecoJetsCaloTower] = (*jet).eta();
		  JetRecoPhiCaloTower[NumRecoJetsCaloTower] = (*jet).phi();
//		  cout<<" Phi, eta gen "<< JetRecoPhiCaloTower[NumRecoJetsCaloTower]<<" "<< JetRecoEtaCaloTower[NumRecoJetsCaloTower]<<endl;
		  JetRecoGenRecType[NumRecoJetsCaloTower] = -1;
		  JetRecoGenPartonType[NumRecoJetsCaloTower] = -1;
		  NumRecoJetsCaloTower++;
		  
		}
	    }
	}
    }
    }

     if( NumRecoJetsCaloTower > 0 && NumGenJets > 0 )
     {
       for(int iii=0; iii<NumRecoJetsCaloTower; iii++)
       {  
         for(int jjj=0; jjj<NumGenJets; jjj++)
         {
              double dphi1 = fabs(JetRecoPhiCaloTower[iii]-JetGenPhi[jjj]);
              if(dphi1 > 4.*atan(1.)) dphi1 = 8.*atan(1.) - dphi1;
              double deta1 = JetRecoEtaCaloTower[iii]-JetGenEta[jjj];
              double dr1 = sqrt(dphi1*dphi1+deta1*deta1);
              if(dr1 < 0.3) {JetRecoGenRecType[iii] = jjj;JetRecoGenPartonType[iii] = JetGenCode[jjj]; 
              cout<<" Associated jet "<< iii<<" "<<JetRecoGenRecType[iii]<<" "<<JetRecoGenPartonType[iii]<<endl;
              cout<<" Etcalo "<<JetRecoEtCaloTower[iii]<<" ETgen "<<JetGenEt[jjj]<<endl;
              } 
         } 
     }
    } 

// JetPlusTrack correction
     NumRecoJetsCorrected = 0;
     {
     edm::Handle<reco::CaloJetCollection> jets;
     iEvent.getByLabel(mInputJetsCorrected, jets);
     if (!jets.isValid()) {
       // can't find it!
       if (!allowMissingInputs_) {cout<<"JetPlusTrack CaloTowers are missed "<<endl; 
	 *jets;  // will throw the proper exception
       }
     } else {
       reco::CaloJetCollection::const_iterator jet = jets->begin ();

       cout<<" Size of jets "<<jets->size()<<endl;
       if(jets->size() > 0 )
	 {
	   for (; jet != jets->end (); jet++)
	     {
	       if( NumRecoJetsCorrected < 100 )
		 {
		   JetRecoEtCorrected[NumRecoJetsCorrected] = (*jet).et();
		   JetRecoEtaCorrected[NumRecoJetsCorrected] = (*jet).eta();
		   JetRecoPhiCorrected[NumRecoJetsCorrected] = (*jet).phi();
		   
		   if( JetRecoGenRecType[NumRecoJetsCorrected] > -1 )cout<<" Calo jet "<<JetRecoEtCaloTower[NumRecoJetsCorrected]<<" Cor "<<JetRecoEtCorrected[NumRecoJetsCorrected]<<" Gen "<<JetGenEt[(int)(JetRecoGenRecType[NumRecoJetsCorrected])]<<endl;
		   
		   NumRecoJetsCorrected++;
		 }
	     }
	 }
     }
     }


// CaloTowers from RecHits
// Load EcalRecHits

    std::vector<edm::InputTag>::const_iterator i;
    int iecal = 0;
    double empty_jet_energy_ecal = 0.; 


   for(int jjj=0; jjj<NumRecoJetsCaloTower; jjj++)
   {
    JetRecoEtRecHit[jjj] = 0.;

    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
      {
      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);
      if (!ec.isValid()) {
	// can't find it!
	if (!allowMissingInputs_) {cout<<" Ecal rechits are missed "<<endl; 
	  *ec;  // will throw the proper exception
	}
      } else {
	// EcalBarrel = 1, EcalEndcap = 2
	for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
	    recHit != (*ec).end(); ++recHit)
	  {
	    
	    GlobalPoint pos = geo->getPosition(recHit->detid());
	    double deta = pos.eta() - JetRecoEtaCaloTower[jjj];
	    double dphi = fabs(pos.phi() - JetRecoPhiCaloTower[jjj]); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    double dr = sqrt(dphi*dphi + deta*deta);
	    double dphi_empty = fabs(pos.phi()+4.*atan(1.) - JetRecoPhiCaloTower[jjj]);
	    if(dphi_empty > 4.*atan(1.)) dphi_empty = 8.*atan(1.) - dphi_empty;
	    double dr_empty = sqrt(dphi_empty*dphi_empty + deta*deta);
	    
	    
	    if(dr<mCone)
	      {
		//       cout<<" Ecal digis "<<jjj<<" "<<JetRecoEtCaloTower[jjj]<<" "<<JetRecoEtaCaloTower[jjj]<<" "<<JetRecoPhiCaloTower[jjj]<<" "<<(*recHit).energy()<<endl;
		//       cout<<" Ecal detid "<<pos.eta()<<" "<<pos.phi()<<" "<<dr<<" JetRecoEtRecHit[jjj]  "<<JetRecoEtRecHit[jjj]<<endl;
            JetRecoEtRecHit[jjj] = JetRecoEtRecHit[jjj] + (*recHit).energy();
	    //            cout<<" New Ecal energy "<<(*recHit).energy()<<endl;
	      }
	    if(dr_empty<mCone)
	      {
		empty_jet_energy_ecal = empty_jet_energy_ecal + (*recHit).energy();         
	      }
	  }
      }
      }
      iecal++;
    }
//        cout<<" Additional ECAL "<<jjj<<" "<<JetRecoEtRecHit[jjj]<<" Eta "<<JetRecoEtaCaloTower[jjj]<<endl;
   }
// Hcal Barrel and endcap for isolation
   double empty_jet_energy_hcal = 0.;
   {
   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel(hbhelabel_,hbhe);
   if (!hbhe.isValid()) {
     // can't find it!
     cout<<" Exception in hbhe "<<endl;
     if (!allowMissingInputs_) {
       *hbhe;  // will throw the proper exception
     }
   } else {
     for(int jjj=0; jjj<NumRecoJetsCaloTower; jjj++)
       {	 
	 for(HBHERecHitCollection::const_iterator hbheItr = (*hbhe).begin();
	     hbheItr != (*hbhe).end(); ++hbheItr)
	   {
	     DetId id = (hbheItr)->detid();
	     GlobalPoint pos = geo->getPosition(hbheItr->detid());
	     double deta = pos.eta() - JetRecoEtaCaloTower[jjj];
	     double dphi = fabs(pos.phi() - JetRecoPhiCaloTower[jjj]);
	     if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	     double dr = sqrt(dphi*dphi + deta*deta);
	     double dphi_empty = fabs(pos.phi()+4.*atan(1.) - JetRecoPhiCaloTower[jjj]);
	     if(dphi_empty > 4.*atan(1.)) dphi_empty = 8.*atan(1.) - dphi_empty;
	     double dr_empty = sqrt(dphi_empty*dphi_empty + deta*deta);
	     
	     if(dr<mCone)
	       {
		 //           cout<<" HCAL JetRecoEtRecHit[jjj]  "<<JetRecoEtRecHit[jjj]<<endl;
		 JetRecoEtRecHit[jjj] = JetRecoEtRecHit[jjj] + (*hbheItr).energy();
	       }
	     if(dr_empty<mCone)
	       {
		 empty_jet_energy_hcal = empty_jet_energy_hcal + (*hbheItr).energy();
	       }
	   }
//	 cout<<" Additional HCAL energy "<<jjj<<" "<<JetRecoEtRecHit[jjj]<<" "<<JetRecoEtaCaloTower[jjj]<<" "<<JetRecoEtaCaloTower[jjj]<<endl;
       }
   }
   }

//  }
       EcalEmpty[0] = empty_jet_energy_ecal;
       HcalEmpty[0] = empty_jet_energy_hcal;

// Tracker
/*
    edm::Handle<reco::TrackCollection> tracks;
    iEvent.getByLabel("ctfWithMaterialTracks", tracks);

    reco::TrackCollection::const_iterator trk;
    int iTracks = 0;
    for ( trk = tracks->begin(); trk != tracks->end(); ++trk){
      TrackRecoEt[iTracks] = trk->pt();
      TrackRecoEta[iTracks] = trk->eta();
      TrackRecoPhi[iTracks] = trk->phi();
      iTracks++;
    }
    NumRecoTrack = iTracks;
    cout<<" Number of tracks "<<NumRecoTrack<<endl;
*/
  for(int jjj = 0; jjj<NumRecoJetsCaloTower; jjj++)
  {
   if(JetRecoGenPartonType[jjj] > -1){
   cout<<" Calo energy "<<JetRecoEtCaloTower[jjj]<<" RecHit energy "<<JetRecoEtRecHit[jjj]<<" Empty cone ECAL "<<empty_jet_energy_ecal<<
   " Empty cone HCAL "<<empty_jet_energy_hcal <<" association with gen jet "<<JetRecoGenRecType[jjj]
   <<" Association with parton  "<<JetRecoGenPartonType[jjj]<<endl;        
   int i1 = JetRecoGenPartonType[jjj];
   int i2 = JetRecoGenPartonType[jjj];
    cout<<" Parton energy "<<pt[i1]<<" type "<<parton[i2]<<endl; 
   }
  } 
   myTree->Fill();
   
}
} // namespace cms

