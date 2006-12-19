#include <iostream>
//
#include "RecoEgamma/EgammaPhotonProducers/interface/ConvertedPhotonAnalyzer.h"
// 
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/EgammaCandidates/interface/ConvertedPhoton.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
// 
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
// 
#include "FWCore/Framework/interface/MakerMacros.h"
 

using namespace std;

 
ConvertedPhotonAnalyzer::ConvertedPhotonAnalyzer( const edm::ParameterSet& pset )
   : fOutputFileName_( pset.getUntrackedParameter<string>("HistOutFile",std::string("TestConversions.root")) ),
     fOutputFile_(0)
{

  convertedPhotonCollectionProducer_ = pset.getParameter<std::string>("phoProducer");
  convertedPhotonCollection_ = pset.getParameter<std::string>("convertedPhotonCollection");
  //
  conversionTrackCandidateProducer_ = pset.getParameter<std::string>("conversionTrackCandidateProducer");
  outInTrackCandidateCollection_    = pset.getParameter<std::string>("outInTrackCandidateCollection");
  inOutTrackCandidateCollection_    = pset.getParameter<std::string>("inOutTrackCandidateCollection");
  //
  conversionOITrackProducer_ = pset.getParameter<std::string>("conversionOITrackProducer");
  conversionIOTrackProducer_ = pset.getParameter<std::string>("conversionIOTrackProducer");
  

}

void ConvertedPhotonAnalyzer::beginJob( const edm::EventSetup& )
{
 
  fOutputFile_   = new TFile( fOutputFileName_.c_str(), "RECREATE" ) ;
  
  //// All MC photons  
  h_MCphoE_ = new TH1F("MCphoE","MC photon energy",100,0.,100.);
  h_MCphoPt_ = new TH1F("MCphoPt","MC photon pt",100,0.,100.);
  h_MCphoEta_ = new TH1F("MCphoEta","MC photon eta",100,-2.5,2.5);

  //// MC Converted photons
  h_MCConvE_ = new TH1F("MCConvE","MC converted photon energy",100,0.,100.);
  h_MCConvPt_ = new TH1F("MCConvPt","MC converted photon pt",100,0.,100.);
  h_MCConvEta_ = new TH1F("MCConvEta","MC convrted photon eta",100,-2.5,2.5);
  h_MCConvR_ = new TH1F("MCConvR","Conversion radius",100,0.,120.);

  //// Reconstructed Converted photons
  h_scE_ = new TH1F("scE","Uncorrected photons : SC Energy ",100,0., 50.);
  h_scEta_ = new TH1F("scEta","Uncorrected photons:  SC Eta ",40,-3., 3.);
  h_scPhi_ = new TH1F("scPhi","Uncorrected photons: SC Phi ",40,0., 6.28);
  //// Reconstructed OutIn Tracks
  h_OItk_inPt_ = new TH1F("OItkinPt","OutIn Tracks Pt ",100,0., 50.);
  h_OItk_nHits_ = new TH1F("OItknHits","OutIn Tracks Hits ",20,0.5, 19.5);
  //// Reconstructed InOut Tracks
  h_IOtk_inPt_ = new TH1F("IOtkinPt","InOut Tracks Pt ",100,0., 50.);
  h_IOtk_nHits_ = new TH1F("IOtknHits","InOut Tracks Hits ",20,0.5, 19.5);
  
  tree_ = new TTree("MCConvertedPhotons","MC converted photon");
  tree_->Branch("mcPhoEnergy",mcPhoEnergy,"mcPhoEnergy[10]/F");
  tree_->Branch("mcPhoEt",mcPhoEt,"mcPhoEt[10]/F");
  tree_->Branch("mcPhoPt",mcPhoPt,"mcPhoPt[10]/F");
  tree_->Branch("mcPhoEta",mcPhoEta,"mcPhoEta[10]/F");
  tree_->Branch("mcPhoPhi",mcPhoPhi,"mcPhoPhi[10]/F");
  tree_->Branch("mcConvR",mcConvR,"mcConvR[10]/F");
  tree_->Branch("mcConvZ",mcConvZ,"mcConvZ[10]/F");
  tree_->Branch("idTrk1",idTrk1,"idTrk1[10]/I");
  tree_->Branch("idTrk2",idTrk2,"idTrk2[10]/I");
  
  

  for (int i=0; i<10; ++i) {
    mcPhoEnergy[i]=-99.;
    mcPhoEt[i]=-99.;
    mcPhoPt[i]=-99.;
    mcPhoEta[i]=-99.;
    mcPhoPhi[i]=-99.;
    mcConvR[i]=-99;
    mcConvZ[i]=-999;
    idTrk1[i]=-1;
    idTrk2[i]=-1;
    
  }
  
  return ;
}

void ConvertedPhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& )
{
  
  
  using namespace edm;
  
  edm::LogInfo("ConvertedPhotonAnalyzer") << "Analyzing event number: " << e.id() << "\n";
  std::cout << "ConvertedPhotonAnalyzer" << "Analyzing event number: " << e.id() << std::endl;
  
  // Local variables  
  const int SINGLE=1;
  const int DOUBLE=2;
  const int PYTHIA=3;
  const int ELECTRON_FLAV=1;
  const int PIZERO_FLAV=2;
  const int PHOTON_FLAV=3;
  
  int ievtype=0;
  int ievflav=0;
  
  
  std::vector<SimTrack*> photonTracks;
  std::vector<SimTrack*> pizeroTracks;
  std::vector<const SimTrack *> trkFromConversion;
  SimVertex primVtx;   
  vector<int> convInd;

  ///// Get the recontructed converted photons
  Handle<reco::ConvertedPhotonCollection> convertedPhotonHandle; 
  try {
    e.getByLabel(convertedPhotonCollectionProducer_, convertedPhotonCollection_ , convertedPhotonHandle);
  } catch ( cms::Exception& ex ) {
    edm::LogError("ConvertedPhotonAnalyzer") << "Error! can't get collection with label " << convertedPhotonCollection_.c_str() ;
  }
  const reco::ConvertedPhotonCollection phoCollection = *(convertedPhotonHandle.product());
  std::cout << " ConvertedPhotonAnalyze converted photon collection size " << phoCollection.size() << std::endl;


  //// Get the candidate tracks from conversions
  Handle<TrackCandidateCollection> outInTrkCandidateHandle;
  try {
    e.getByLabel(conversionTrackCandidateProducer_,  outInTrackCandidateCollection_ , outInTrkCandidateHandle);
  }  catch ( cms::Exception& ex ) {
    edm::LogError("ConvertedPhotonAnalyzer") << "Error! can't get collection with label " <<  outInTrackCandidateCollection_.c_str() ;
  }

  std::cout << " ConvertedPhotonAnalyzer outInTrackCandidate collection size " << (*outInTrkCandidateHandle).size() << std::endl;

  Handle<TrackCandidateCollection> inOutTrkCandidateHandle;
  try {
    e.getByLabel(conversionTrackCandidateProducer_,  inOutTrackCandidateCollection_ , inOutTrkCandidateHandle);
  }  catch ( cms::Exception& ex ) {
    edm::LogError("ConvertedPhotonAnalyzer") << "Error! can't get collection with label " <<  inOutTrackCandidateCollection_.c_str() ;
  }

  std::cout << " ConvertedPhotonAnalyzer inOutTrackCandidate collection size " << (*inOutTrkCandidateHandle).size() << std::endl;


  //// Get the CKF tracks from conversions
  Handle<reco::TrackCollection> outInTrkHandle;
  try {
    e.getByLabel(conversionOITrackProducer_,  outInTrkHandle);
  }  catch ( cms::Exception& ex ) {
    edm::LogError("ConvertedPhotonAnalyzer") << "Error! can't get collection with label " <<  outInTrackCollection_.c_str() ;
  }

  std::cout << " ConvertedPhotonAnalyzer outInTrack collection size " << (*outInTrkHandle).size() << std::endl;
 // Loop over Out In Tracks
  for( reco::TrackCollection::const_iterator  iTk =  (*outInTrkHandle).begin(); iTk !=  (*outInTrkHandle).end(); iTk++) {
    std::cout << "  ConvertedPhotonAnalyzer Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << std::endl;  

    h_OItk_inPt_->Fill ( sqrt(iTk->innerMomentum().Mag2()) );
    h_OItk_nHits_->Fill ( iTk->recHitsSize() );
   
    std::cout << "  ConvertedPhotonAnalyzer Out In Track Extra inner momentum  " << iTk->extra()->outerMomentum() << std::endl;  
   
  }



  Handle<reco::TrackCollection> inOutTrkHandle;
  try {
    e.getByLabel(conversionIOTrackProducer_, inOutTrkHandle);
  }  catch ( cms::Exception& ex ) {
    edm::LogError("ConvertedPhotonAnalyzer") << "Error! can't get collection with label " <<  inOutTrackCollection_.c_str() ;
  }

  std::cout << " ConvertedPhotonAnalyzer inOutTrack collection size " << (*inOutTrkHandle).size() << std::endl;



 // Loop over In Out Tracks
  for( reco::TrackCollection::const_iterator  iTk =  (*inOutTrkHandle).begin(); iTk !=  (*inOutTrkHandle).end(); iTk++) {
    std::cout << "  ConvertedPhotonAnalyzer In Out Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << std::endl;  

    h_IOtk_inPt_->Fill  ( sqrt (iTk->innerMomentum().Mag2() ) );
    h_IOtk_nHits_->Fill ( iTk->recHitsSize() );
   
    std::cout << "  ConvertedPhotonAnalyzer In Out  Track Extra inner momentum  " << iTk->extra()->outerMomentum() << std::endl;  

  }

  std::cout << "  ConvertedPhotonAnalyzer Starting loop over photon candidates " << std::endl;
  // Loop over ConvertedPhoton candidates 
  for( reco::ConvertedPhotonCollection::const_iterator  iPho = phoCollection.begin(); iPho != phoCollection.end(); iPho++) {
    std::cout << "  ConvertedPhotonAnalyzer SC energy " << (*iPho).superCluster()->energy() <<  std::endl;
    h_scE_->Fill( (*iPho).superCluster()->energy() );
    h_scEta_->Fill( (*iPho).superCluster()->position().eta() );
    h_scPhi_->Fill( (*iPho).superCluster()->position().phi() );
    
  }

 
  //////////////////// Get the MC truth   
  std::cout << " ConvertedPhotonAnalyzer Looking for MC truth " << std::endl;
  edm::Handle<edm::HepMCProduct> HepMCEvt;
  //e.getByLabel("VtxSmeared", "", HepMCEvt);
  e.getByLabel("source", "", HepMCEvt);
  
  //   const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();
  
  
   //get simtrack info
   std::vector<SimTrack> theSimTracks;
   std::vector<SimVertex> theSimVertices;

   edm::Handle<SimTrackContainer> SimTk;
   edm::Handle<SimVertexContainer> SimVtx;
   e.getByLabel("g4SimHits",SimTk);
   e.getByLabel("g4SimHits",SimVtx);

   theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
   theSimVertices.insert(theSimVertices.end(),SimVtx->begin(),SimVtx->end());

   fill(theSimTracks,  theSimVertices);
   
   cout << " ConvertedPhotonAnalyzer::analyze This Event has " <<  theSimTracks.size() << " sim tracks " << endl;
   cout << " ConvertedPhotonAnalyzer::analyze This Event has " <<  theSimVertices.size() << " sim vertices " << endl;
   if (  ! theSimTracks.size() ) cout << " ConvertedPhotonAnalyzer::analyze Event number " << e.id() << " has NO sim tracks " << endl;

   int iPV=-1;   
   int partType1=0;
   int partType2=0;
   std::vector<SimTrack>::iterator iFirstSimTk = theSimTracks.begin();
   if (  !(*iFirstSimTk).noVertex() ) {
     iPV =  (*iFirstSimTk).vertIndex();

     int vtxId =   (*iFirstSimTk).vertIndex();
     primVtx = theSimVertices[vtxId];
    
     
     
     partType1 = (*iFirstSimTk).type();
     cout <<  "ConvertedPhotonAnalyzer::analyze Primary vertex id " << iPV << " first track type " << (*iFirstSimTk).type() << endl;  
   } else {
     cout << " ConvertedPhotonAnalyzer::analyze First track has no vertex " << endl;
   
   }

   // Look at a second track
   iFirstSimTk++;
   if ( iFirstSimTk!=  theSimTracks.end() ) {
     
     if (  (*iFirstSimTk).vertIndex() == iPV) {
       partType2 = (*iFirstSimTk).type();  
       cout <<  "ConvertedPhotonAnalyzer::analyze second track type " << (*iFirstSimTk).type() << " vertex " <<  (*iFirstSimTk).vertIndex() << endl;  
    
     } else {
       cout << "ConvertedPhotonAnalyzer::analyze Only one kine track from Primary Vertex " << endl;
     }
   }

   cout << " Loop over all particles " << endl;

   int npv=0;
   int iPho=0;
   int iPizero=0;
   //   theSimTracks.reset();
   for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
     if (  (*iSimTk).noVertex() ) continue;
     cout << " Particle type " <<  (*iSimTk).type() << " Sim Track ID " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  endl;  
     if ( (*iSimTk).vertIndex() == iPV ) {
       npv++;
       if ( (*iSimTk).type() == 22) {
	 cout << " Found a primary photon with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  endl; 
	 convInd.push_back(0);
        
         photonTracks.push_back( &(*iSimTk) );

	 
	 CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();
	 h_MCphoPt_->Fill(  momentum.perp());
	 h_MCphoEta_->Fill( momentum.pseudoRapidity());
	 h_MCphoE_->Fill( momentum.rho());


         if ( iPho < 10) {
           
	   mcPhoEnergy[iPho]= momentum.e(); 
	   mcPhoPt[iPho]= momentum.perp(); 
	   mcPhoEt[iPho]= momentum.et(); 
	   mcPhoEta[iPho]= momentum.pseudoRapidity();
	   mcPhoPhi[iPho]= momentum.phi();

	 }

	 iPho++;

       } else if ( (*iSimTk).type() == 11 || (*iSimTk).type()==-11 ) {
	 cout << " Found a primary electron with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  endl;



       } else if ( (*iSimTk).type() == 111 ) {
	 cout << " Found a primary pi0 with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  endl;	 


         pizeroTracks.push_back( &(*iSimTk) );
	 
	 
	 CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();
	 
	 if ( iPizero < 10) {

	   mcPizEnergy[iPizero]= momentum.e(); 
	   mcPizPt[iPizero]= momentum.perp(); 
	   mcPizEt[iPizero]= momentum.et(); 
	   mcPizEta[iPizero]= momentum.pseudoRapidity();
	   mcPizPhi[iPizero]= momentum.phi();

	 }

	 iPizero++;



	 
       }


     }

   } 

   cout << "ConvertedPhotonAnalyzer::analyze There are " << npv << " particles originating in the PV " << endl;
     
   if(npv > 4) {
     ievtype = PYTHIA;
   } else if(npv == 1) {
     if( abs(partType1) == 11 ) {
        ievtype = SINGLE; ievflav = ELECTRON_FLAV;
     } else if(partType1 == 111) {
       ievtype = SINGLE; ievflav = PIZERO_FLAV;
     } else if(partType1 == 22) {
       ievtype = SINGLE; ievflav = PHOTON_FLAV;
     }
   } else if(npv == 2) {
     if (  abs(partType1) == 11 && abs(partType2) == 11 ) {
       ievtype = DOUBLE; ievflav = ELECTRON_FLAV;
     } else if(partType1 == 111 && partType2 == 111)   {
       ievtype = DOUBLE; ievflav = PIZERO_FLAV;
     } else if(partType1 == 22 && partType2 == 22)   {
       ievtype = DOUBLE; ievflav = PHOTON_FLAV;
     }
   }      
   
   //////  Look into converted photons  

   if(ievflav == PHOTON_FLAV) {

     cout << " It's a primary PHOTON event with " << photonTracks.size() << endl;

     int nConv=0;
     int iConv=0;
     iPho=0;
     for (std::vector<SimTrack*>::iterator iPhoTk = photonTracks.begin(); iPhoTk != photonTracks.end(); ++iPhoTk){
       cout << "ConvertedPhotonAnalyzer::analyze Looping on the primary gamma looking for conversions " << (*iPhoTk)->momentum() << " photon track ID " << (*iPhoTk)->trackId() << endl;

       for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
	 if (  (*iSimTk).noVertex() )                    continue;
	 if ( (*iSimTk).vertIndex() == iPV )             continue; 
         if ( abs((*iSimTk).type()) != 11  )             continue;

	 int vertexId = (*iSimTk).vertIndex();
	 SimVertex vertex = theSimVertices[vertexId];
         int motherId=-1;
	

         cout << " Secondary from photons particle type " << (*iSimTk).type() << " trackId " <<  (*iSimTk).trackId() << " vertex ID " << vertexId << endl;
         if ( vertex.parentIndex()  ) {

	   unsigned  motherGeantId = vertex.parentIndex(); 
	   std::map<unsigned, unsigned >::iterator association = geantToIndex_.find( motherGeantId );
	   if(association != geantToIndex_.end() )
	     motherId = association->second;
	   
	   
	   int motherType = motherId == -1 ? 0 : theSimTracks[motherId].type();
	   
	   cout << " Parent to this vertex   motherId " << motherId << " mother type " <<  motherType << " Sim track ID " <<  theSimTracks[motherId].trackId() << endl;
           
	   if ( theSimTracks[motherId].trackId() == (*iPhoTk)->trackId() ) {
	     cout << " Found the Mother Photon " << endl;
             /// store this electron since it's from a converted photon
	     trkFromConversion.push_back(&(*iSimTk ) );
	   }
 
           


	 } else {
	   cout << " This vertex has no parent tracks " <<  endl;
	 }
	 
	 
       } // End of loop over the SimTracks      
       
       if ( trkFromConversion.size() > 0 ) {
      
	 nConv++;
	 convInd[iPho]=nConv;         

	 int convVtxId =  trkFromConversion[0]->vertIndex();
	 SimVertex convVtx = theSimVertices[convVtxId];
	 CLHEP::HepLorentzVector vtxPosition = convVtx.position();

	 CLHEP::HepLorentzVector momentum = (*iPhoTk)->momentum();
	 h_MCConvPt_->Fill(  momentum.perp());
	 h_MCConvEta_->Fill( momentum.pseudoRapidity());
	 h_MCConvE_->Fill( momentum.rho());
	 h_MCConvR_->Fill ( vtxPosition.perp()/10. ) ;

         if ( nConv <= 10) {         
	   
	   mcConvR[iConv]=vtxPosition.perp()/10. ;
	   mcConvZ[iConv]=vtxPosition.z()/10. ;
	   
	   if ( trkFromConversion.size() > 1) {
	     idTrk1[iConv]= trkFromConversion[0]->trackId();
	     idTrk2[iConv]= trkFromConversion[1]->trackId();
	    
	   } else {
	     idTrk1[iConv]=trkFromConversion[0]->trackId();
	     idTrk2[iConv]=-1;
	   }
	   
	 }
	 
         iConv++;
       }
       
       iPho++;       
     } // End loop over the primary photons
     
     
   }   // Event with one or two photons 



   if(ievflav == PIZERO_FLAV) {
     cout << " It's a primary Pi0 event with " << pizeroTracks.size() << endl;

     iPho=0;     
     for (std::vector<SimTrack*>::iterator iPizTk = pizeroTracks.begin(); iPizTk != pizeroTracks.end(); ++iPizTk){
       cout << "ConvertedPhotonAnalyzer::analyze Looping on the primary Pi0 looking for decay products " << (*iPizTk)->momentum() << " Pi0 track ID " << (*iPizTk)->trackId() << endl;
       
       for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
	 if ( (*iSimTk).noVertex()     )                    continue;
	 if ( (*iSimTk).vertIndex() == iPV )            continue; 
	 
	 int vertexId = (*iSimTk).vertIndex();
	 SimVertex vertex = theSimVertices[vertexId];
	 HepLorentzVector vtxshift = vertex.position() - primVtx.position();
	 if  ( vtxshift.vect().mag()/10. < 0.1 && vtxshift.t() < 1.e-9 ) {
	   ////  Photons or direct Dalitz from pizero
	   CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();

	   if ( (*iSimTk).type() ==22 ) {
	     if ( iPho < 10) {
	       
	       mcPhoEnergy[iPho]= momentum.e(); 
	       mcPhoPt[iPho]= momentum.perp(); 
	       mcPhoEt[iPho]= momentum.et(); 
	       mcPhoEta[iPho]= momentum.pseudoRapidity();
	       mcPhoPhi[iPho]= momentum.phi();
	       
	     }
	     iPho++;
	   } else if (  abs((*iSimTk).type()) == 11  ) {
	     /// Fill in what's missing
           
	   }


             

	 } else {

	   /// Converted photons from pizero

	 }




       }
     }



     
   }   // Event with one or two Pi0







   tree_->Fill();
  
}




void ConvertedPhotonAnalyzer::endJob()
{
       
   fOutputFile_->Write() ;
   fOutputFile_->Close() ;
   cout << " ConvertedPhotonAnalyzer::endJob " << endl;
   
   return ;
}
 

void  ConvertedPhotonAnalyzer::fill(const std::vector<SimTrack>& simTracks, const std::vector<SimVertex>& simVertices) {

  // Watch out there ! A SimVertex is in mm (stupid), 
 
  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  // Empty event, do nothin'
  if ( nVtx == 0 ) return;

  // create a map associating geant particle id and position in the 
  // event SimTrack vector
  for( unsigned it=0; it<nTks; ++it ) {
    geantToIndex_[ simTracks[it].trackId() ] = it;
    cout << " ConvertedPhotonAnalyzer::fill it " << it << " simTracks[it].trackId() " <<  simTracks[it].trackId() << endl;
 
  }  



}

