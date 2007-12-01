#include "JetMETCorrections/JetPlusTrack/src/JetPlusTrackAnalysis.h"

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
   mInputJetsCaloTower = iConfig.getParameter<edm::InputTag>("src1");
   mInputJetsGen = iConfig.getParameter<edm::InputTag>("src2");			  
   mInputJetsCorrected = iConfig.getParameter<edm::InputTag>("src3");
   m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks");

   hbhelabel_ = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
   holabel_ = iConfig.getParameter<edm::InputTag>("HORecHitCollectionLabel");
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
   allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",false);

   cout<<" JetPlusTrack constructor "<<endl;			  
}

JetPlusTrackAnalysis::~JetPlusTrackAnalysis()
{
    cout<<" JetPlusTrack destructor "<<endl;
}

void JetPlusTrackAnalysis::beginJob( const edm::EventSetup& iSetup)
{
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
   myTree->Branch("JetRecoEtCaloTower",  JetRecoEtCaloTower, "JetRecoEtCaloTower[10]/F");
   myTree->Branch("JetRecoEtaCaloTower",  JetRecoEtaCaloTower, "JetRecoEtaCaloTower[10]/F");
   myTree->Branch("JetRecoPhiCaloTower",  JetRecoPhiCaloTower, "JetRecoPhiCaloTower[10]/F");
   myTree->Branch("JetRecoEtRecHit",  JetRecoEtRecHit, "JetRecoEtRecHit[10]/F");
   myTree->Branch("JetRecoGenRecType", JetRecoGenRecType, "JetRecoGenRecType[10]/F");
   myTree->Branch("JetRecoGenPartonType", JetRecoGenPartonType , "JetRecoGenPartonType[10]/F");
   myTree->Branch("EcalEmpty", EcalEmpty , "EcalEmpty[10]/F");
   myTree->Branch("HcalEmpty", HcalEmpty , "HcalEmpty[10]/F");
//
   myTree->Branch("NumRecoJetsCorrected", &NumRecoJetsCorrected, "NumRecoJetsCorrected/I");
   myTree->Branch("JetRecoEtCorrected",  JetRecoEtCorrected, "JetRecoEtCorrected[10]/F");
   myTree->Branch("JetRecoEtCorrectedZS",  JetRecoEtCorrectedZS, "JetRecoEtCorrectedZS[10]/F");
   myTree->Branch("JetRecoEtaCorrected",  JetRecoEtaCorrected, "JetRecoEtaCorrected[10]/F");
   myTree->Branch("JetRecoPhiCorrected",  JetRecoPhiCorrected, "JetRecoPhiCorrected[10]/F");
// GenJet block
   myTree->Branch("NumGenJets", &NumGenJets, "NumGenJets/I");
   myTree->Branch("JetGenEt",  JetGenEt, "JetGenEt[10]/F");
   myTree->Branch("JetGenEta",  JetGenEta, "JetGenEta[10]/F");
   myTree->Branch("JetGenPhi",  JetGenPhi, "JetGenPhi[10]/F");
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
  std::vector<edm::Provenance const*> theProvenance;
  iEvent.getAllProvenance(theProvenance);
  for( std::vector<edm::Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }



   run = iEvent.id().run();
   event = iEvent.id().event();
//
//  Rememeber parton
//
  float pt[2],eta[2],phi[2];
  int parton[2];
  int tagparton;
  try {
  cout<<" Try to take HepMCProduct "<<endl;
    vector< edm::Handle< edm::HepMCProduct > > EvtHandles ;
    iEvent.getManyByType( EvtHandles ) ;
   
   for ( unsigned int i=0; i<EvtHandles.size(); i++ )
   {

      if ( EvtHandles[i].isValid() )
      {

         const HepMC::GenEvent* Evt = EvtHandles[i]->GetEvent() ;
         HepMC::GenEvent::vertex_const_iterator Vtx = Evt->vertices_begin() ;

        int ihep = 0; 
         for (HepMC::GenEvent::particle_const_iterator
            Part = Evt->particles_begin() ; Part!=Evt->particles_end(); Part++ )
         {


         if ( EvtHandles[i].provenance()->moduleLabel() == "VtxSmeared" )
            {

             cout<<" particle "<<(*Part)->pdg_id()<<" "<<(*Part)->status()<<" "<<((*Part)->momentum()).perp()<<endl;
//     " charge "<<(*Part)->particledata().charge()<<endl;
//             int mychar = (*Part)->particledata().charge();
             if(ihep <2)
             {
                pt[ihep] = ((*Part)->momentum()).perp();
                eta[ihep] = ((*Part)->momentum()).eta();
                phi[ihep] = ((*Part)->momentum()).phi();
                parton[ihep] = (*Part)->pdg_id();
             } 
             Code[ihep] = (*Part)->pdg_id();
//             Charge[ihep] = mychar;
             partpx[ihep] = (*Part)->momentum().px();
             partpy[ihep] = (*Part)->momentum().py();
             partpz[ihep] = (*Part)->momentum().pz();
             parte[ihep] = (*Part)->momentum().e();
             partm[ihep] = (*Part)->momentum().m();
             ihep++;
             NumPart = ihep;
   
            }
            else
            {
             cout<<" Label "<<EvtHandles[i].provenance()->moduleLabel()<<endl;
            }

         }
      }
   }
  } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) {cout<<" GenParticles are missed "<<endl; throw e;}
  }
//  Generated jet
   NumGenJets = 0;
   int icode = -1;
     try {
        edm::Handle<reco::GenJetCollection> jets;
        iEvent.getByLabel(mInputJetsGen, jets);
       reco::GenJetCollection::const_iterator jet = jets->begin ();
       if(jets->size() > 0 )
       {
         for (; jet != jets->end (); jet++)
         {
            if( NumGenJets < 1 )
            {
              JetGenEt[NumGenJets] = (*jet).et();
              JetGenEta[NumGenJets] = (*jet).eta();
              JetGenPhi[NumGenJets] = (*jet).phi();
               cout<<" Phi, eta gen "<< JetGenPhi[NumGenJets]<<" "<< JetGenEta[NumGenJets]<<endl;
              NumGenJets++;
// Find the parton
              double dphi1 = fabs((*jet).phi()-phi[0]);
              if(dphi1 > 4.*atan(1.)) dphi1 = 8.*atan(1.) - dphi1;
              double dphi2 = fabs((*jet).phi()-phi[1]);
              if(dphi2 > 4.*atan(1.)) dphi2 = 8.*atan(1.) - dphi2;
              double deta1 = (*jet).eta()-eta[0];
              double deta2 = (*jet).eta()-eta[1];
              double dr1 = sqrt(dphi1*dphi1+deta1*deta1);
              double dr2 = sqrt(dphi2*dphi2+deta2*deta2); 
              if(dr1 > 0.5 && dr2 > 0.5) cout<<" Fake jet "<<endl;
              if(dr1 < 0.5) icode = 0;
              if(dr2 < 0.5) icode = 1;
  
            }

         }
       }
     } catch (std::exception& e) { // can't find it!
       if (!allowMissingInputs_) throw e;
     }


// CaloJets

    NumRecoJetsCaloTower = 0;
    int icodegenrec = -1;
     try {
        edm::Handle<reco::CaloJetCollection> jets;
        iEvent.getByLabel(mInputJetsCaloTower, jets);
        reco::CaloJetCollection::const_iterator jet = jets->begin ();

        cout<<" Size of jets "<<jets->size()<<endl;

       if(jets->size() > 0 )
       {
         for (; jet != jets->end (); jet++)
         {

            if( NumRecoJetsCaloTower < 1 )
            {

// Association with gen jet

              double dphi1 = fabs((*jet).phi()-JetGenPhi[0]);
              if(dphi1 > 4.*atan(1.)) dphi1 = 8.*atan(1.) - dphi1;
              double deta1 = (*jet).eta()-JetGenEta[0];
              double dr1 = sqrt(dphi1*dphi1+deta1*deta1);
              if(dr1 < 0.5) icodegenrec = 1;

             JetRecoEtCaloTower[NumRecoJetsCaloTower] = (*jet).et();
             JetRecoEtaCaloTower[NumRecoJetsCaloTower] = (*jet).eta();
             JetRecoPhiCaloTower[NumRecoJetsCaloTower] = (*jet).phi();
               cout<<" Phi, eta gen "<< JetRecoPhiCaloTower[NumRecoJetsCaloTower]<<" "<< JetRecoEtaCaloTower[NumRecoJetsCaloTower]<<endl;
             JetRecoGenRecType[NumRecoJetsCaloTower] = icodegenrec;
             JetRecoGenPartonType[NumRecoJetsCaloTower] = icode;
             NumRecoJetsCaloTower++;

            }
         }
       }
     } catch (std::exception& e) { // can't find it!
       if (!allowMissingInputs_) {cout<<"CaloTowers are missed "<<endl; throw e;}
     }
// JetPlusTrack correction
     NumRecoJetsCorrected = 0;
     try {
        edm::Handle<reco::CaloJetCollection> jets;
        iEvent.getByLabel(mInputJetsCorrected, jets);
        reco::CaloJetCollection::const_iterator jet = jets->begin ();

        cout<<" Size of jets "<<jets->size()<<endl;
       if(jets->size() > 0 )
       {
         for (; jet != jets->end (); jet++)
         {
            if( NumRecoJetsCorrected < 1 )
            {
             JetRecoEtCorrected[NumRecoJetsCorrected] = (*jet).et();
             JetRecoEtaCorrected[NumRecoJetsCorrected] = (*jet).eta();
             JetRecoPhiCorrected[NumRecoJetsCorrected] = (*jet).phi();

// Zero-suupression correction
             double theta = 2.*atan(exp(-1.*(*jet).eta()));
             float detjpt = JetRecoEtCorrected[0]/sin(theta)-JetRecoEtCaloTower[0]/sin(theta);
             float jetetnew = JetRecoEtCaloTower[0]/(1.+0.2018-(52.71/(JetRecoEtCaloTower[0]+100.)));  
             float jetenew = jetetnew/sin(theta);
             float jetetcor = (jetenew+detjpt)*sin(theta); 

             JetRecoEtCorrectedZS[NumRecoJetsCorrected] = jetetcor;

             cout<<" delta "<<detjpt<<" "<<jetetnew<<" "<<theta<<" "<<sin(theta)<<" "<<jetenew<<endl;
             cout<<" Calo jet "<<JetRecoEtCaloTower[0]<<" Cor "<<JetRecoEtCorrected[0]<<" +ZScor "<<jetetcor<<" Gen "<<JetGenEt[0]<<endl;

             NumRecoJetsCorrected++;

            }
         }
       }

      } catch (std::exception& e) { // can't find it!
       if (!allowMissingInputs_) {cout<<"JetPlusTrack CaloTowers are missed "<<endl; throw e;}
      }
     


// CaloTowers from RecHits
// Load EcalRecHits

    std::vector<edm::InputTag>::const_iterator i;
    int iecal = 0;
    double jet_energy = 0.;
    double empty_jet_energy_ecal = 0.; 
    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
    try {

      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);

       for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
                                                recHit != (*ec).end(); ++recHit)
       {
// EcalBarrel = 1, EcalEndcap = 2

         GlobalPoint pos = geo->getPosition(recHit->detid());
         double deta = pos.eta() - JetRecoEtaCaloTower[0];
         double dphi = fabs(pos.phi() - JetRecoPhiCaloTower[0]); 
         if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
         double dr = sqrt(dphi*dphi + deta*deta);
//          cout<<" Ecal digis "<<iecal<<" "<<(*recHit).energy()<<endl;
         double dphi_empty = fabs(pos.phi()+4.*atan(1.) - JetRecoPhiCaloTower[0]);
         if(dphi_empty > 4.*atan(1.)) dphi_empty = 8.*atan(1.) - dphi_empty;
         double dr_empty = sqrt(dphi_empty*dphi_empty + deta*deta);


         if(dr<0.5)
         {
         jet_energy = jet_energy + (*recHit).energy();
         }
         if(dr_empty<0.5)
         {
         empty_jet_energy_ecal = empty_jet_energy_ecal + (*recHit).energy();
         
         }

       }

    } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) {cout<<" Ecal rechits are missed "<<endl; throw e;}
    }
     iecal++;
    }

// Hcal Barrel and endcap for isolation
   double empty_jet_energy_hcal = 0.;
    try {
      edm::Handle<HBHERecHitCollection> hbhe;
      iEvent.getByLabel(hbhelabel_,hbhe);

  for(HBHERecHitCollection::const_iterator hbheItr = (*hbhe).begin();
                                           hbheItr != (*hbhe).end(); ++hbheItr)
      {
        DetId id = (hbheItr)->detid();
        GlobalPoint pos = geo->getPosition(hbheItr->detid());
         double deta = pos.eta() - JetRecoEtaCaloTower[0];
         double dphi = fabs(pos.phi() - JetRecoPhiCaloTower[0]);
         if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
         double dr = sqrt(dphi*dphi + deta*deta);
         double dphi_empty = fabs(pos.phi()+4.*atan(1.) - JetRecoPhiCaloTower[0]);
         if(dphi_empty > 4.*atan(1.)) dphi_empty = 8.*atan(1.) - dphi_empty;
         double dr_empty = sqrt(dphi_empty*dphi_empty + deta*deta);


         if(dr<0.5)
         {
         jet_energy = jet_energy + (*hbheItr).energy();
         }
         if(dr_empty<0.5)
         {
         empty_jet_energy_hcal = empty_jet_energy_hcal + (*hbheItr).energy();
         }


      }
    } catch (std::exception& iEvent) { // can't find it!
      cout<<" Exception in hbhe "<<endl;
      if (!allowMissingInputs_) throw iEvent;
    }
//  }
       JetRecoEtRecHit[0] = jet_energy;
       EcalEmpty[0] = empty_jet_energy_ecal;
       HcalEmpty[0] = empty_jet_energy_hcal;

// Tracker
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


   cout<<" Calo energy "<<JetRecoEtCaloTower[0]<<" RecHit energy "<<jet_energy<<" Empty cone ECAL "<<empty_jet_energy_ecal<<
   " Empty cone HCAL "<<empty_jet_energy_hcal <<" association with gen jet "<<JetRecoGenRecType[0]
   <<" Association with parton  "<<JetRecoGenPartonType[0]<<endl;        
   if( JetRecoGenRecType[0] > 0 ) cout<<" Gen energy "<< JetGenEt[0]<<endl;
   int i1 = JetRecoGenPartonType[0];
    int i2 = JetRecoGenPartonType[0];
   if( JetRecoGenPartonType[0] > -1) cout<<" Parton energy "<<pt[i1]<<" type "<<parton[i2]<<endl; 
   myTree->Fill();
}
} // namespace cms
