#include "RecoJets/JetPlusTracks/test/PatJetPlusTrackAnalysis.h"

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/JetReco/interface/JetID.h"

#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

#include "DataFormats/PatCandidates/interface/Jet.h"



using namespace std;
namespace cms
{

PatJetPlusTrackAnalysis::PatJetPlusTrackAnalysis(const edm::ParameterSet& iConfig):centrality_(0)
{
   cout<<" Start PatJetPlusTrackAnalysis now"<<endl;
   
   mCone05 = iConfig.getParameter<double>("Cone1");
   
   mCone07 = iConfig.getParameter<double>("Cone2");
   
   data = iConfig.getParameter<int>("Data");
   
   mInputJetsCaloTower = iConfig.getParameter<edm::InputTag>("src1");

   mInputJetsCaloTower2 = iConfig.getParameter<edm::InputTag>("src11");
   
   mInputJetsGen = iConfig.getParameter<edm::InputTag>("src2");	
   
   mInputJetsGen2 = iConfig.getParameter<edm::InputTag>("src22");
   	
   mJetsIDName            = iConfig.getParameter<std::string>("jetsID");
   
   mJetsIDName2            = iConfig.getParameter<std::string>("jetsID2");
   
   mInputJetsCorrected = iConfig.getParameter<edm::InputTag>("src3");

   mInputJetsCorrected2 = iConfig.getParameter<edm::InputTag>("src4");
   

   hbhelabel_ = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
   
   holabel_ = iConfig.getParameter<edm::InputTag>("HORecHitCollectionLabel");
   
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   
   fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
  
   allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",false);
   cout<<" PatJetPlusTrackAnalysis constructor::data "<<data<<endl;			  
}

PatJetPlusTrackAnalysis::~PatJetPlusTrackAnalysis()
{
    cout<<" JetPlusTrack destructor "<<endl;
}

void PatJetPlusTrackAnalysis::beginJob()
{

   cout<<" Begin job "<<endl;

   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
   myTree = new TTree("JetPlusTrack","JetPlusTrack Tree");
   myTree->Branch("run",  &run, "run/I");
   myTree->Branch("event",  &event, "event/I");
   myTree->Branch("NvtxEv",  &NvtxEv, "NvtxEv/I");
   myTree->Branch("Ntrkv",  &Ntrkv, "Ntrkv/I");
   myTree->Branch("VertNDF",  &VertNDF, "VertNDF/I");
   myTree->Branch("VertChi2",  &VertChi2, "VertChi2/F");
   myTree->Branch("centrality_bin",  &centrality_bin, "centraloty_bin/I");

   NumRecoJetsCaloTower = 0;
   NumRecoJetsJPTCorrected = 0;
   NumRecoJetsJPTCorrected2 = 0;
   NumRecoJetsRecHit = 0;
   NumGenJets = 0;
   NumGenJets2 = 0;
   NumPart = 0;
   NvtxEv = 0;
   Ntrkv = 0;

// Jet Reco CaloTower
   myTree->Branch("NumRecoJetsCaloTower", &NumRecoJetsCaloTower, "NumRecoJetsCaloTower/I");
   myTree->Branch("JetRecoEtCaloTower",  JetRecoEtCaloTower, "JetRecoEtCaloTower[10]/F");
   myTree->Branch("JetRecoEtaCaloTower",  JetRecoEtaCaloTower, "JetRecoEtaCaloTower[10]/F");
   myTree->Branch("JetRecoPhiCaloTower",  JetRecoPhiCaloTower, "JetRecoPhiCaloTower[10]/F");
   myTree->Branch("JetRecoEtRecHit05",  JetRecoEtRecHit05, "JetRecoEtRecHit05[10]/F");
   myTree->Branch("JetRecoEmf",  JetRecoEmf, "JetRecoEmf[10]/F");
   myTree->Branch("JetRecofHPD",  JetRecofHPD, "JetRecofHPD[10]/F");
   myTree->Branch("JetRecofRBX",  JetRecofRBX, "JetRecofRBX[10]/F");


   myTree->Branch("JetRecoGenRecType", JetRecoGenRecType, "JetRecoGenRecType[10]/F");
   myTree->Branch("JetRecoGenPartonType", JetRecoGenPartonType , "JetRecoGenPartonType[10]/F");

// Pileup part

   myTree->Branch("EnergyCaloTowerEtaPlus", EnergyCaloTowerEtaPlus , "EnergyCaloTowerEtaPlus[42]/F");
   myTree->Branch("EnergyCaloTowerEtaMinus", EnergyCaloTowerEtaMinus , "EnergyCaloTowerEtaMinus[42]/F");

//   
   myTree->Branch("NumRecoJetsCaloTower2", &NumRecoJetsCaloTower2, "NumRecoJetsCaloTower2/I");
   myTree->Branch("JetRecoEtCaloTower2",  JetRecoEtCaloTower2, "JetRecoEtCaloTower2[10]/F");
   myTree->Branch("JetRecoEtRecHit07",  JetRecoEtRecHit07, "JetRecoEtRecHit07[10]/F");
   myTree->Branch("JetRecoEtaCaloTower2",  JetRecoEtaCaloTower2, "JetRecoEtaCaloTower2[10]/F");
   myTree->Branch("JetRecoPhiCaloTower2",  JetRecoPhiCaloTower2, "JetRecoPhiCaloTower2[10]/F");
   myTree->Branch("JetRecoEmf2",  JetRecoEmf2, "JetRecoEmf2[10]/F");
   myTree->Branch("JetRecofHPD2",  JetRecofHPD2, "JetRecofHPD2[10]/F");
   myTree->Branch("JetRecofRBX2",  JetRecofRBX2, "JetRecofRBX2[10]/F");
   
   
//
   myTree->Branch("NumRecoJetsJPTCorrected", &NumRecoJetsJPTCorrected, "NumRecoJetsJPTCorrected/I");
   myTree->Branch("JetRecoEtJPTCorrected",  JetRecoEtJPTCorrected, "JetRecoEtJPTCorrected[10]/F");
   myTree->Branch("JetRecoEtZSPCorrected", JetRecoEtZSPCorrected, "JetRecoEtZSPCorrected[10]/F");
   myTree->Branch("JetRecoEtCaloJetInit",JetRecoEtCaloJetInit , "JetRecoEtCaloJetInit[10]/F");
   myTree->Branch("JetRecoEtaCaloJetInit",JetRecoEtaCaloJetInit , "JetRecoEtaCaloJetInit[10]/F");
   myTree->Branch("JetRecoPhiCaloJetInit",JetRecoPhiCaloJetInit , "JetRecoPhiCaloJetInit[10]/F");
   myTree->Branch("JetRecoEtaJPTCorrected",  JetRecoEtaJPTCorrected, "JetRecoEtaJPTCorrected[10]/F");
   myTree->Branch("JetRecoPhiJPTCorrected",  JetRecoPhiJPTCorrected, "JetRecoPhiJPTCorrected[10]/F");
   myTree->Branch("JetRecoInitEmf",  JetRecoInitEmf, "JetRecoInitEmf[10]/F");
   myTree->Branch("JetRecoInitfHPD",  JetRecoInitfHPD, "JetRecoInitfHPD[10]/F");
   myTree->Branch("JetRecoInitfRBX",  JetRecoInitfRBX, "JetRecoInitfRBX[10]/F");
   myTree->Branch("JetRecoInitMN90a", JetRecoInitMN90a , "JetRecoInitMN90a[10]/F");
   myTree->Branch("JetRecoInitMN90Hits",  JetRecoInitMN90Hits, "JetRecoInitMN90Hits[10]/F");
   myTree->Branch("JetRecoJPTSumETrack",  JetRecoJPTSumETrack, "JetRecoJPTSumETrack[10]/F");
   myTree->Branch("JetRecoJPTTrackMultInVertInCalo",  JetRecoJPTTrackMultInVertInCalo, "JetRecoJPTTrackMultInVertInCalo[10]/I");
   myTree->Branch("JetRecoJPTTrackMultInVertOutCalo",  JetRecoJPTTrackMultInVertOutCalo, "JetRecoJPTTrackMultInVertOutCalo[10]/I");
   myTree->Branch("JetRecoJPTTrackMultOutVertInCalo",  JetRecoJPTTrackMultOutVertInCalo, "JetRecoJPTTrackMultOutVertInCalo[10]/I");
   
   
//
   myTree->Branch("NumRecoJetsJPTCorrected2", &NumRecoJetsJPTCorrected2, "NumRecoJetsJPTCorrected2/I");
   myTree->Branch("JetRecoEtJPTCorrected2",  JetRecoEtJPTCorrected2, "JetRecoEtJPTCorrected2[10]/F");
   myTree->Branch("JetRecoEtZSPCorrected2", JetRecoEtZSPCorrected2, "JetRecoEtZSPCorrected2[10]/F");
   myTree->Branch("JetRecoEtCaloJetInit2",JetRecoEtCaloJetInit2 , "JetRecoEtCaloJetInit2[10]/F");
   myTree->Branch("JetRecoEtaCaloJetInit2",JetRecoEtaCaloJetInit2 , "JetRecoEtaCaloJetInit2[10]/F");
   myTree->Branch("JetRecoPhiCaloJetInit2",JetRecoPhiCaloJetInit2 , "JetRecoPhiCaloJetInit2[10]/F");
   myTree->Branch("JetRecoEtaJPTCorrected2",  JetRecoEtaJPTCorrected2, "JetRecoEtaJPTCorrected2[10]/F");
   myTree->Branch("JetRecoPhiJPTCorrected2",  JetRecoPhiJPTCorrected2, "JetRecoPhiJPTCorrected2[10]/F");
   myTree->Branch("JetRecoInitEmf2",  JetRecoInitEmf2, "JetRecoInitEmf2[10]/F");
   myTree->Branch("JetRecoInitfHPD2",  JetRecoInitfHPD2, "JetRecoInitfHPD2[10]/F");
   myTree->Branch("JetRecoInitfRBX2",  JetRecoInitfRBX2, "JetRecoInitfRBX2[10]/F");
   myTree->Branch("JetRecoInitMN90a2", JetRecoInitMN90a2 , "JetRecoInitMN90a2[10]/F");
   myTree->Branch("JetRecoInitMN90Hits2",  JetRecoInitMN90Hits2, "JetRecoInitMN90Hits2[10]/F");
   myTree->Branch("JetRecoJPTSumETrack2",  JetRecoJPTSumETrack2, "JetRecoJPTSumETrack2[10]/F");
   myTree->Branch("JetRecoJPTTrackMultInVertInCalo2",  JetRecoJPTTrackMultInVertInCalo2, "JetRecoJPTTrackMultInVertInCalo2[10]/I");
   myTree->Branch("JetRecoJPTTrackMultInVertOutCalo2",  JetRecoJPTTrackMultInVertOutCalo2, "JetRecoJPTTrackMultInVertOutCalo2[10]/I");
   myTree->Branch("JetRecoJPTTrackMultOutVertInCalo2",  JetRecoJPTTrackMultOutVertInCalo2, "JetRecoJPTTrackMultOutVertInCalo2[10]/I");

   
// GenJet block
   myTree->Branch("NumGenJets", &NumGenJets, "NumGenJets/I");
   myTree->Branch("JetGenEt",    JetGenEt, "JetGenEt[10]/F");
   myTree->Branch("JetGenEta",   JetGenEta, "JetGenEta[10]/F");
   myTree->Branch("JetGenPhi",   JetGenPhi, "JetGenPhi[10]/F");
   myTree->Branch("JetGenCode",  JetGenCode, "JetGenCode[10]/I");
   
// GenJet block
   myTree->Branch("NumGenJets2", &NumGenJets2, "NumGenJets2/I");
   myTree->Branch("JetGenEt2",  JetGenEt2, "JetGenEt2[10]/F");
   myTree->Branch("JetGenEta2",  JetGenEta2, "JetGenEta2[10]/F");
   myTree->Branch("JetGenPhi2",  JetGenPhi2, "JetGenPhi2[10]/F");
   myTree->Branch("JetGenCode2",  JetGenCode2, "JetGenCode2[10]/I");
   
   
// Particle block
   myTree->Branch("NumPart", &NumPart, "NumPart/I");
   myTree->Branch("Code",  Code, "Code[2]/I");
   myTree->Branch("Charge",  Charge, "Charge[2]/I");
   myTree->Branch("partpx",  partpx, "partpx[2]/F");
   myTree->Branch("partpy",  partpy, "partpy[2]/F");
   myTree->Branch("partpz",  partpz, "partpz[2]/F");
   myTree->Branch("parte",  parte, "parte[2]/F");
   myTree->Branch("partm",  partm, "partm[2]/F");  

}

void PatJetPlusTrackAnalysis::beginRun(edm::Run const&, edm::EventSetup const& iSetup) 
{
// Calo Geometry
}

void PatJetPlusTrackAnalysis::endJob()
{

   cout << "===== Start writing user histograms =====" << endl;
   hOutputFile->SetCompressionLevel(2);
   hOutputFile->cd();
   myTree->Write();
   hOutputFile->Close() ;
   cout << "===== End writing user histograms =======" << endl;
   
}

void PatJetPlusTrackAnalysis::analyze(
                                         const edm::Event& iEvent,
                                         const edm::EventSetup& theEventSetup)  
{
    cout<<" JetPlusTrack analyze "<<endl;
/*
  std::vector<edm::Provenance const*> theProvenance;
  iEvent.getAllProvenance(theProvenance);
  for( std::vector<edm::Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }
*/
   edm::ESHandle<CaloGeometry> pG;
   theEventSetup.get<CaloGeometryRecord>().get(pG);
   geo = pG.product();


   run = iEvent.id().run();
   event = iEvent.id().event();
//
//  Rememeber parton
//
  float pt[2],eta[2],phi[2];
  int parton[2];
  //  int tagparton;
  
  if( data == 0 ) {
  
//   cout<<" Try to take HepMCProduct "<<endl;
//   edm::Handle< edm::HepMCProduct >  EvtHandles ;
//   iEvent.getByType( EvtHandles ) ;

// //NR==================================================

//   if (!EvtHandles.isValid()) {
//     // can't find it!
//     if (!allowMissingInputs_) {cout<<" GenParticles are missed "<<endl;}
//     *EvtHandles;  // will throw the proper exception
//   } else {
//          const HepMC::GenEvent* Evt = EvtHandles->GetEvent() ;

//         int ihep = 0; 
         
//          for (HepMC::GenEvent::particle_const_iterator
//             Part = Evt->particles_begin() ; Part!=Evt->particles_end(); Part++ )
//          {
//              if(ihep == 6 || ihep == 7)
//              {
//                 cout<<" parton "<<(*Part)->pdg_id()<<" "<<(*Part)->status()<<" "<<((*Part)->momentum()).perp()<<endl;
//                 pt[ihep-6] = ((*Part)->momentum()).perp();
//                 eta[ihep-6] = ((*Part)->momentum()).eta();
//                 phi[ihep-6] = ((*Part)->momentum()).phi();
//                 parton[ihep-6] = (*Part)->pdg_id();
//              } 
// //             Code[ihep] = (*Part)->pdg_id();
// //             partpx[ihep] = (*Part)->momentum().px();
// //             partpy[ihep] = (*Part)->momentum().py();
// //             partpz[ihep] = (*Part)->momentum().pz();
// //             parte[ihep] = (*Part)->momentum().e();
// //             partm[ihep] = (*Part)->momentum().m();
//              ihep++;
// //             NumPart = ihep;
//          }
//   }
//   NumPart = 2;
  
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
	     if( NumGenJets < 10 )
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
		 icode = -1;
		 if(dr1 < 0.5 || dr2 < 0.5) {
		   if(dr1 < 0.5) icode = 0;
		   if(dr2 < 0.5) icode = 1;
		 } 
		   JetGenEt[NumGenJets] = (*jet).et();
		   JetGenEta[NumGenJets] = (*jet).eta();
		   JetGenPhi[NumGenJets] = (*jet).phi();
		   JetGenCode[NumGenJets] = icode;

                   cout<<" Gen jet "<<NumGenJets<<" "<<JetGenEt[NumGenJets]<<" "<<JetGenEta[NumGenJets]<<" "<<JetGenPhi[NumGenJets]<<" "<<JetGenCode[NumGenJets]<<endl;

		   NumGenJets++;
	       }
	   }
       }
   }
   }
//  Generated jet iterative cone
   NumGenJets2 = 0;
   icode = -1;
   {
   edm::Handle<reco::GenJetCollection> jets;
   iEvent.getByLabel(mInputJetsGen2, jets);
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
	     if( NumGenJets2 < 10 )
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
		 icode = -1;
		 if(dr1 < 0.5 || dr2 < 0.5) {
		   if(dr1 < 0.5) icode = 0;
		   if(dr2 < 0.5) icode = 1;
		 } 
		   JetGenEt2[NumGenJets2] = (*jet).et();
		   JetGenEta2[NumGenJets2] = (*jet).eta();
		   JetGenPhi2[NumGenJets2] = (*jet).phi();
		   JetGenCode2[NumGenJets2] = icode;
		   NumGenJets2++;
	       }
	   }
       }
   }
   }
     if(NumGenJets == 0) return;     
     if(NumGenJets2 == 0) return;
} 
// Clean events

// Check if Primary vertex exist
    int bx = iEvent.bunchCrossing();
//    if(bx != 51 && bx != 2724 ) {std::cout<<" Event with bad bunchcrossing? "<<bx<<std::endl;} 

   edm::Handle<reco::VertexCollection> pvHandle; 
   //iEvent.getByLabel("offlinePrimaryVertices",pvHandle);
   iEvent.getByLabel("hiSelectedVertex",pvHandle);
   const reco::VertexCollection & vertices = *pvHandle.product();
   bool result = false;   

   NvtxEv = 0; 
   Ntrkv = 0;

   for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it)
   {
      if(!(*it).isFake()) {
      
      std::cout<<" Track size "<<it->tracksSize()<<" "<<fabs(it->z())<<" "<<fabs(it->position().rho())<<std::endl;
      
      if(it->tracksSize() > 0 && 
         ( fabs(it->z()) <= 15. ) &&
         ( fabs(it->position().rho()) <= 2. )
       ) 
         {
	     result = true;
             NvtxEv++;
	     if(NvtxEv == 1) {
	     VertNDF = (*it).ndof();
	     VertChi2 = (*it).chi2();
             reco::Vertex::trackRef_iterator ittrk;
             for(ittrk =(*it).tracks_begin();ittrk != (*it).tracks_end(); ++ittrk)
               if( (*it).trackWeight(*ittrk)>0.5 ) Ntrkv++;	       
	     } // First vertex
         } // if for vertex
      } // nonfake vertex
   } // cycle on vertex

   if(!result) { std::cout<<" Vertex is outside 15 cms "<<std::endl; return;}
   if( NvtxEv > 1 ) {std::cout<<" More then one vertex "<<std::endl; return;}
      

   std::cout<<" PV exists in the acceptable range (+-15 cm) and bx = "<<bx<<std::endl;  

// Add centrality

    if(!centrality_) centrality_ = new CentralityProvider(theEventSetup);
    centrality_->newEvent(iEvent,theEventSetup); 
    double c = centrality_->centralityValue();
    centrality_bin = centrality_->getBin();
    
    std::cout<<" Centrality bin "<<centrality_bin<<std::endl;

/*    
    edm::Handle<reco::Centrality> centrality;
    iEvent.getByLabel ("hiCentrality",centrality);
    
    cbins_ = getCentralityBinsFromDB(theEventSetup);

    double hf_ = centrality->EtHFhitSum();
    double sumET_ = centrality->EtMidRapiditySum();
    int bin_ = cbins_->getBin(hf_);

*/    

//    std::cout<<" Centrality bin "<<centrality_bin<<std::endl;
//
// CaloJets
//
//      edm::Handle<edm::ValueMap<reco::JetID> > jetsID;
//      iEvent.getByLabel(mJetsIDName,jetsID);
//      edm::Handle<edm::ValueMap<reco::JetID> > jetsID2;
//      iEvent.getByLabel(mJetsIDName2,jetsID2);



    NumRecoJetsCaloTower = 0;
    {
    
    int ind=0;
    
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
	      
	      if( NumRecoJetsCaloTower < 10 )
		{
		  
		  // Association with gen jet
		  edm::RefToBase<reco::Jet> jetRef(edm::Ref<reco::CaloJetCollection>(jets,ind));
		  
		  JetRecoEtCaloTower[NumRecoJetsCaloTower] = (*jet).et();
		  JetRecoEtaCaloTower[NumRecoJetsCaloTower] = (*jet).eta();
		  JetRecoPhiCaloTower[NumRecoJetsCaloTower] = (*jet).phi();
		  JetRecoGenRecType[NumRecoJetsCaloTower] = -1;
		  JetRecoGenPartonType[NumRecoJetsCaloTower] = -1;
		  
                  JetRecoEmf[NumRecoJetsCaloTower] = (*jet).emEnergyFraction();
		  //                  JetRecofHPD[NumRecoJetsCaloTower] = (*jetsID)[jetRef].fHPD;
		  // JetRecofRBX[NumRecoJetsCaloTower] = (*jetsID)[jetRef].fRBX;
		  
		  std::cout<<" Calo jet "<<NumRecoJetsCaloTower<<" "<<JetRecoEtCaloTower[NumRecoJetsCaloTower]<<" "<<JetRecoEtaCaloTower[NumRecoJetsCaloTower]<<" "<<JetRecoPhiCaloTower[NumRecoJetsCaloTower]<<std::endl; 
		  ind++ ;
		  NumRecoJetsCaloTower++;
		}
	    }
	}
    }
    }
    
    std::cout<<" Before 2"<<std::endl;
    
    NumRecoJetsCaloTower2 = 0;
    {
    edm::Handle<reco::CaloJetCollection> jets;
    int ind=0;

    iEvent.getByLabel(mInputJetsCaloTower2, jets);
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
	      
	      if( NumRecoJetsCaloTower2 < 10 )
		{
		  
		  // Association with gen jet
		  edm::RefToBase<reco::Jet> jetRef(edm::Ref<reco::CaloJetCollection>(jets,ind));
		  JetRecoEtCaloTower2[NumRecoJetsCaloTower2] = (*jet).et();
		  JetRecoEtaCaloTower2[NumRecoJetsCaloTower2] = (*jet).eta();
		  JetRecoPhiCaloTower2[NumRecoJetsCaloTower2] = (*jet).phi();
		  
		  
                  JetRecoEmf2[NumRecoJetsCaloTower2] = (*jet).emEnergyFraction();
		  //     JetRecofHPD2[NumRecoJetsCaloTower2] = (*jetsID2)[jetRef].fHPD;
		  //  JetRecofRBX2[NumRecoJetsCaloTower2] = (*jetsID2)[jetRef].fRBX;
		  
		  ind++ ;
		  NumRecoJetsCaloTower2++;
		  
		  
		}
	    }
	}
    }
    }
//    
// If no gen or calo jets of one of the collection - do nothing
//
          
    std::cout<<" Before 3"<<std::endl;
    
// JetPlusTrack correction
     NumRecoJetsJPTCorrected = 0;
     {
//     edm::Handle<reco::CaloJetCollection> jets;
       //edm::Handle<reco::JPTJetCollection> jets;
     edm::Handle<std::vector<pat::Jet> > jets;
     

     iEvent.getByLabel(mInputJetsCorrected, jets);
     if (!jets.isValid()) {
       // can't find it!
       if (!allowMissingInputs_) {cout<<"JetPlusTrack CaloTowers are missed "<<endl; 
	 *jets;  // will throw the proper exception
       }
     } else {
     //  reco::CaloJetCollection::const_iterator jet = jets->begin ();
       //reco::JPTJetCollection::const_iterator jet = jets->begin ();
       std::vector<pat::Jet>::const_iterator patjet = jets->begin ();
       cout<<" Size of jets "<<jets->size()<<endl;
       if(jets->size() > 0 )
	 {
	   for (; patjet != jets->end (); patjet++)
	     {
	       const reco::JPTJet *jet = dynamic_cast<const reco::JPTJet*>((*patjet).originalObject());
	       if( NumRecoJetsJPTCorrected < 10 )
		 {
		   JetRecoEtJPTCorrected[NumRecoJetsJPTCorrected] = (*patjet).et();
		   JetRecoEtaJPTCorrected[NumRecoJetsJPTCorrected] = (*patjet).eta();
		   JetRecoPhiJPTCorrected[NumRecoJetsJPTCorrected] = (*patjet).phi();
// Look to the CaloJet initiated
		   
		   JetRecoEtCaloJetInit[NumRecoJetsJPTCorrected] = (*jet).getCaloJetRef()->et();
		   JetRecoEtaCaloJetInit[NumRecoJetsJPTCorrected] = (*jet).getCaloJetRef()->eta();
		   JetRecoPhiCaloJetInit[NumRecoJetsJPTCorrected] = (*jet).getCaloJetRef()->phi();
		   
		   edm::RefToBase<reco::Jet> jptjetRef = jet->getCaloJetRef();
                   reco::CaloJet const * rawcalojet = dynamic_cast<reco::CaloJet const *>( &* jptjetRef);
		   		   
//                   JetRecoInitEmf[NumRecoJetsJPTCorrected] = rawcalojet->emEnergyFraction();
		   //    JetRecoInitfHPD[NumRecoJetsJPTCorrected] = (*jetsID)[(*jet).getCaloJetRef()].fHPD;
		   //  JetRecoInitfRBX[NumRecoJetsJPTCorrected] = (*jetsID)[(*jet).getCaloJetRef()].fRBX;
//		   JetRecoInitMN90a[NumRecoJetsJPTCorrected] = rawcalojet->n90();
		   //  JetRecoInitMN90Hits[NumRecoJetsJPTCorrected] = (*jetsID)[(*jet).getCaloJetRef()].n90Hits;
    std::cout<<" PAT JPT jet "<<NumRecoJetsJPTCorrected<<" "<<JetRecoEtJPTCorrected[NumRecoJetsJPTCorrected]<<" "<<JetRecoEtaJPTCorrected[NumRecoJetsJPTCorrected]<<" "<<JetRecoPhiJPTCorrected[NumRecoJetsJPTCorrected]<<" Calo "<<JetRecoEtCaloJetInit[NumRecoJetsJPTCorrected]<<" "<<
JetRecoEtaCaloJetInit[NumRecoJetsJPTCorrected]<<" "<<JetRecoPhiCaloJetInit[NumRecoJetsJPTCorrected]<<std::endl;		  
		   
// ZSP corrected jet		   
		  JetRecoEtZSPCorrected[NumRecoJetsJPTCorrected] = (*jet).getCaloJetRef()->et()*(*jet).getZSPCor();
// Tracks
/*
	  float px = 0.;
	  float py = 0.;	
	  float pz = 0.;	
          float en = 0.;
	    
          const reco::TrackRefVector pioninin = (*jet).getPionsInVertexInCalo();
	  for(reco::TrackRefVector::const_iterator it = pioninin.begin(); it != pioninin.end(); it++) {
  //            std::cout<<" Track in in "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<std::endl;
	      px+=(*it)->px();
	      py+=(*it)->py();
	      pz+=(*it)->pz();
	      en+=sqrt((*it)->p()*(*it)->p()+0.14*0.14);
          }
	  
          const reco::TrackRefVector pioninout = (*jet).getPionsInVertexOutCalo();
	  for(reco::TrackRefVector::const_iterator it = pioninout.begin(); it != pioninout.end(); it++) {	      
  //            std::cout<<" Track in out "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<std::endl;
	      px+=(*it)->px();
	      py+=(*it)->py();
	      pz+=(*it)->pz();
	      en+=sqrt((*it)->p()*(*it)->p()+0.14*0.14);
          }
	  
           JetRecoJPTTrackMultInVertInCalo[NumRecoJetsJPTCorrected] = pioninin.size();
           JetRecoJPTTrackMultInVertOutCalo[NumRecoJetsJPTCorrected] = pioninout.size();
           JetRecoJPTTrackMultOutVertInCalo[NumRecoJetsJPTCorrected] = (*jet).getPionsOutVertexInCalo().size();
	   
	   
           JetRecoJPTSumETrack[NumRecoJetsJPTCorrected] = sqrt(px*px+py*py+pz*pz) * sin((*jet).theta());
  //         std::cout<<" Jet PT "<<(*jet).et()<<" eta "<<(*jet).eta()<<" Charged "<<JetRecoJPTSumETrack[NumRecoJetsJPTCorrected]<<" "<<en<<" "<<sqrt(px*px+py*py+pz*pz)<< " "<<sin((*jet).theta())<<std::endl;

*/		   
				     
		   
		   NumRecoJetsJPTCorrected++;
		 }
	     }
	 }
     }
     }
     std::cout<<" Before 4"<<std::endl;
// Next jets
      NumRecoJetsJPTCorrected2 = 0;
     {
//     edm::Handle<reco::CaloJetCollection> jets;
       edm::Handle<reco::JPTJetCollection> jets;
       //  edm::Handle<std::vector<pat::Jet> > jets;
 
     iEvent.getByLabel(mInputJetsCorrected2, jets);
     if (!jets.isValid()) {
       // can't find it!
       if (!allowMissingInputs_) {cout<<"JetPlusTrack CaloTowers are missed "<<endl; 
	 *jets;  // will throw the proper exception
       }
     } else {
     //  reco::CaloJetCollection::const_iterator jet = jets->begin ();
       reco::JPTJetCollection::const_iterator jet = jets->begin ();
       //std::vector<pat::Jet>::const_iterator patjet = jets->begin ();
 
       cout<<" Size of jets "<<jets->size()<<endl;
       if(jets->size() > 0 )
	 {
	   for (; jet != jets->end (); jet++)
	     {
               if( NumRecoJetsJPTCorrected2 < 10 )
                 {
		   JetRecoEtJPTCorrected2[NumRecoJetsJPTCorrected2] = (*jet).et();
		   JetRecoEtaJPTCorrected2[NumRecoJetsJPTCorrected2] = (*jet).eta();
		   JetRecoPhiJPTCorrected2[NumRecoJetsJPTCorrected2] = (*jet).phi();
// Look to the CaloJet initiated		   
		   JetRecoEtCaloJetInit2[NumRecoJetsJPTCorrected2] = (*jet).getCaloJetRef()->et();
		   JetRecoEtaCaloJetInit2[NumRecoJetsJPTCorrected2] = (*jet).getCaloJetRef()->eta();
		   JetRecoPhiCaloJetInit2[NumRecoJetsJPTCorrected2] = (*jet).getCaloJetRef()->phi();
		   
		   edm::RefToBase<reco::Jet> jptjetRef = jet->getCaloJetRef();
                   reco::CaloJet const * rawcalojet = dynamic_cast<reco::CaloJet const *>( &* jptjetRef);
		   
                   JetRecoInitEmf2[NumRecoJetsJPTCorrected2] = rawcalojet->emEnergyFraction();
//                   JetRecoInitfHPD2[NumRecoJetsJPTCorrected2] = (*jetsID2)[(*jet).getCaloJetRef()].fHPD;
//                   JetRecoInitfRBX2[NumRecoJetsJPTCorrected2] = (*jetsID2)[(*jet).getCaloJetRef()].fRBX;
		   JetRecoInitMN90a2[NumRecoJetsJPTCorrected2] = rawcalojet->n90();
//		   JetRecoInitMN90Hits2[NumRecoJetsJPTCorrected2] = (*jetsID2)[(*jet).getCaloJetRef()].n90Hits;
		   
		   
// ZSP corrected jet		   
		  JetRecoEtZSPCorrected2[NumRecoJetsJPTCorrected2] = (*jet).getCaloJetRef()->et()*(*jet).getZSPCor();


// Tracks
  if(centrality_bin == 0 ) {
	  float px = 0.;
	  float py = 0.;	
	  float pz = 0.;	
	  float en = 0.;	
	    
          const reco::TrackRefVector pioninin = (*jet).getPionsInVertexInCalo();
	  for(reco::TrackRefVector::const_iterator it = pioninin.begin(); it != pioninin.end(); it++) {
 //             std::cout<<"PatJetPlusTrackAnalysis, Track in in "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()
 //                      <<" "<<(*it)->phi()<<std::endl;
	      px+=(*it)->px();
	      py+=(*it)->py();
	      pz+=(*it)->pz();
	      en+=sqrt((*it)->p()*(*it)->p()+0.14*0.14);
          }
	  
          const reco::TrackRefVector pioninout = (*jet).getPionsInVertexOutCalo();
	  for(reco::TrackRefVector::const_iterator it = pioninout.begin(); it != pioninout.end(); it++) {	      
//              std::cout<<" PatJetPlusTrackAnalysis, Track in out "<<(*it)->p()<<" "<<(*it)->pt()<<" "<<(*it)->eta()
//                       <<" "<<(*it)->phi()<<std::endl;
	      px+=(*it)->px();
	      py+=(*it)->py();
	      pz+=(*it)->pz();
	      en+=sqrt((*it)->p()*(*it)->p()+0.14*0.14);
          }
	  
           JetRecoJPTTrackMultInVertInCalo2[NumRecoJetsJPTCorrected] = pioninin.size();
           JetRecoJPTTrackMultInVertOutCalo2[NumRecoJetsJPTCorrected] = pioninout.size();
           JetRecoJPTTrackMultOutVertInCalo2[NumRecoJetsJPTCorrected] = (*jet).getPionsOutVertexInCalo().size();
           JetRecoJPTSumETrack2[NumRecoJetsJPTCorrected2] = sqrt(px*px+py*py+pz*pz) * sin((*jet).theta());
}
    std::cout<<" RECO JPT jet "<<NumRecoJetsJPTCorrected2<<" "<<JetRecoEtJPTCorrected2[NumRecoJetsJPTCorrected2]<<" "<<
    JetRecoEtaJPTCorrected2[NumRecoJetsJPTCorrected2]<<" "<<JetRecoPhiJPTCorrected2[NumRecoJetsJPTCorrected2]<<
    " Calo "<<JetRecoEtCaloJetInit2[NumRecoJetsJPTCorrected2]<<" "<<
    JetRecoEtaCaloJetInit2[NumRecoJetsJPTCorrected2]<<" "<<JetRecoPhiCaloJetInit2[NumRecoJetsJPTCorrected2]<<std::endl;		  

//    std::cout<<" RECO JPT jet "<<NumRecoJetsJPTCorrected2
//             <<" in in "<<JetRecoJPTTrackMultInVertInCalo2[NumRecoJetsJPTCorrected]
//             <<" in out "<<JetRecoJPTTrackMultInVertOutCalo2[NumRecoJetsJPTCorrected]
//             <<" out in "<<JetRecoJPTTrackMultOutVertInCalo2[NumRecoJetsJPTCorrected]<<std::endl;
		   
		   NumRecoJetsJPTCorrected2++;
		 }
	     }
	 }
     }
     }

// Get tracks



   myTree->Fill();
   
}
} // namespace cms

// define this class as a plugin
#include "FWCore/Framework/interface/MakerMacros.h"
using namespace cms;
DEFINE_FWK_MODULE(PatJetPlusTrackAnalysis);
