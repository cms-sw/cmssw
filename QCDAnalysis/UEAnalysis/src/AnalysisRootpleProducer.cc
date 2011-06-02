// Authors: F. Ambroglini, L. Fano', F. Bechtel
#include <QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducer.h>
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"



//using namespace edm;
//using namespace std;
//using namespace reco;

class GreaterPt{
public:
  bool operator()( const math::XYZTLorentzVector& a, const math::XYZTLorentzVector& b) {
    return a.pt() > b.pt();
  }
};

class GenJetSort{
public:
  bool operator()(const reco::GenJet& a, const reco::GenJet& b) {
    return a.pt() > b.pt();
  }
};

class BasicJetSort{
public:
  bool operator()(const reco::TrackJet& a, const reco::TrackJet& b) {
    return a.pt() > b.pt();
  }
};

class CaloJetSort{
public:
  bool operator()(const reco::CaloJet& a, const reco::CaloJet& b) {
    return a.pt() > b.pt();
  }
};




 
void AnalysisRootpleProducer::store(){

  AnalysisTree->Fill();

}

void AnalysisRootpleProducer::fillEventInfo(int e){
  EventKind = e;
}


AnalysisRootpleProducer::AnalysisRootpleProducer( const edm::ParameterSet& pset )
{
  // flag to ignore gen-level analysis
 
  onlyRECO = pset.getParameter<bool>("OnlyRECO");

  // particle, track and jet collections
  mcEvent             = pset.getParameter<edm::InputTag>( "MCEvent"                   );
  genJetCollName      = pset.getParameter<edm::InputTag>( "GenJetCollectionName"      );
  chgJetCollName      = pset.getParameter<edm::InputTag>( "ChgGenJetCollectionName"   );
  tracksJetCollName   = pset.getParameter<edm::InputTag>( "TracksJetCollectionName"   );
  recoCaloJetCollName = pset.getParameter<edm::InputTag>( "RecoCaloJetCollectionName" );
  chgGenPartCollName  = pset.getParameter<edm::InputTag>( "ChgGenPartCollectionName"  );
  tracksCollName      = pset.getParameter<edm::InputTag>( "TracksCollectionName"      );
  genEventScaleTag    = pset.getParameter<edm::InputTag>( "genEventScale"             );

  // trigger results
  triggerResultsTag = pset.getParameter<edm::InputTag>("triggerResults");
  triggerEventTag   = pset.getParameter<edm::InputTag>("triggerEvent"  );

  piG = acos(-1.);
  pdgidList.reserve(200);
}

void AnalysisRootpleProducer::beginJob()
{
 
  // used TFileService for output to root file

 AnalysisTree = fs->make<TTree>("AnalysisTree","MBUE Analysis Tree ");

  AnalysisTree->Branch("EventKind",&EventKind,"EventKind/I");

  // save TClonesArrays of TLorentzVectors
  // i.e. store 4-vectors of particles and jets
  
 
  MonteCarlo = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("MonteCarlo", "TClonesArray", &MonteCarlo, 128000, 0);

  Track = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("Track", "TClonesArray", &Track, 128000, 0);

  InclusiveJet = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("InclusiveJet", "TClonesArray", &InclusiveJet, 128000, 0);

  ChargedJet = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("ChargedJet", "TClonesArray", &ChargedJet, 128000, 0);

  TracksJet = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("TracksJet", "TClonesArray", &TracksJet, 128000, 0);

  CalorimeterJet = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("CalorimeterJet", "TClonesArray", &CalorimeterJet, 128000, 0);

  acceptedTriggers = new TClonesArray("TObjString", 10000);
  AnalysisTree->Branch("acceptedTriggers", "TClonesArray", &acceptedTriggers, 128000, 0);

  AnalysisTree->Branch("genEventScale", &genEventScale, "genEventScale/D");
 
  AnalysisTree->Branch("eventNum",&eventNum,"eventNum/I");
  AnalysisTree->Branch("lumiBlock",&lumiBlock,"lumiBlock/I");
  AnalysisTree->Branch("runNumber",&runNumber,"runNumber/I");
  AnalysisTree->Branch("bx",&bx,"bx/I");


}

  
void AnalysisRootpleProducer::analyze( const edm::Event& e, const edm::EventSetup& )
{
  ///
  /// Pythia: genEventScaleTag = "genEventScale"
  /// Herwig: genEventScaleTag = "genEventKTValue"
  ///

  // if ( e.getByLabel( genEventScaleTag, genEventScaleHandle ) ) genEventScale = *genEventScaleHandle;
 
  eventNum   = e.id().event() ;
  runNumber  = e.id().run() ;
  lumiBlock  = e.luminosityBlock() ;
  bx = e.bunchCrossing();
  if(!onlyRECO){
    edm::Handle<GenEventInfoProduct> hEventInfo;
  e.getByLabel(genEventScaleTag , hEventInfo);
 if (hEventInfo->binningValues().size() > 0)
   { 
    genEventScale = hEventInfo->binningValues()[0];
   }


  }

 
 // access trigger bits by TriggerResults
  edm::Handle<edm::TriggerResults> hltresults;
  try{e.getByLabel(edm::InputTag("TriggerResults::HLT"), hltresults);}
  //try{iEvent.getManyByType(hltresults);}
  catch(...){std::cout<<"The HLT Trigger branch was not correctly taken"<<std::endl;}

  if (e.getByLabel(edm::InputTag("TriggerResults::HLT"), hltresults) )
      {
	edm::TriggerNames const& triggerNames = e.triggerNames(*hltresults);
	acceptedTriggers->Clear();
	int iAcceptedTriggers( 0 ); 
	 int n_TriggerResults( hltresults.product()->size() );
	 //	cout<< n_TriggerResults<<"dimensione"<<endl;
	for (int itrig = 0; itrig != n_TriggerResults; ++itrig){
	  std::string trigName = triggerNames.triggerName(itrig);
	  bool accept = hltresults->accept((const unsigned int )itrig);
	  //	         cout << "HLT " << itrig << "  " << trigName << "accettato"<<accept <<endl;
	  if (accept){
	 new((*acceptedTriggers)[iAcceptedTriggers]) TObjString( trigName.c_str() );
	 ++iAcceptedTriggers;
	  }
	  //else {HLT[itrig] = "NULL";}
     }
	
	
      }
    
    
    
  // gen level analysis
  // skipped, if onlyRECO flag set to true
  
  if(!onlyRECO){
    
    e.getByLabel( mcEvent           , EvtHandle        );
    e.getByLabel( chgGenPartCollName, CandHandleMC     );
    e.getByLabel( chgJetCollName    , ChgGenJetsHandle );
    e.getByLabel( genJetCollName    , GenJetsHandle    );



  ///-------------------------------
    const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
    
    EventKind = Evt->signal_process_id();

    std::vector<math::XYZTLorentzVector> GenPart;
    std::vector<reco::GenJet> ChgGenJetContainer;
    std::vector<reco::GenJet> GenJetContainer;


    
    GenPart.clear();
 
    ChgGenJetContainer.clear();
    GenJetContainer.clear();
    MonteCarlo->Clear();
    InclusiveJet->Clear();
    ChargedJet->Clear();
    //  Parton->Clear();
    // jets from charged particles at hadron level
    //	std::cout<<"ChgGenJet"<<std::endl; 
   if (ChgGenJetsHandle->size()){

      for ( reco::GenJetCollection::const_iterator it(ChgGenJetsHandle->begin()), itEnd(ChgGenJetsHandle->end());
	    it!=itEnd; ++it)
	{
	  ChgGenJetContainer.push_back(*it);
	}

      std::stable_sort(ChgGenJetContainer.begin(),ChgGenJetContainer.end(),GenJetSort());

      std::vector<reco::GenJet>::const_iterator it(ChgGenJetContainer.begin()), itEnd(ChgGenJetContainer.end());
      for ( int iChargedJet(0); it != itEnd; ++it, ++iChargedJet)
	{
	  new((*ChargedJet)[iChargedJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	}
    }


    // GenJets
   //std::cout<<"GenJet"<<std::endl; 
   
 if (GenJetsHandle->size()){

   for ( reco::GenJetCollection::const_iterator it(GenJetsHandle->begin()), itEnd(GenJetsHandle->end());
	    it!=itEnd; ++it )
	{
	  GenJetContainer.push_back(*it);

	  // 	  Jet::Constituents constituents( (*it).getJetConstituents() );
	  // 	  //cout << "get " << constituents.size() << " constituents" << endl;
	  // 	  for (int iJC(0); iJC<constituents.size(); ++iJC )
	  // 	    {
	  // 	      //cout << "[" << iJC << "] constituent pT = " << constituents[iJC]->pt() << endl;
	  // 	      if (constituents[iJC]->et()<0.5)
	  // 		{
	  // 		  cout << "ERROR!!! [" << iJC << "] constituent pT = " << constituents[iJC]->pt() << endl;
	  // 		}
	  // 	    }
	}

      std::stable_sort(GenJetContainer.begin(),GenJetContainer.end(),GenJetSort());

      std::vector<reco::GenJet>::const_iterator it(GenJetContainer.begin()), itEnd(GenJetContainer.end());
      for ( int iInclusiveJet(0); it != itEnd; ++it, ++iInclusiveJet)
	{
	  new((*InclusiveJet)[iInclusiveJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	}
    }


    // hadron level particles
 // std::cout<<"Particle"<<std::endl;   
  if (CandHandleMC->size()){
    
      for (std::vector<reco::GenParticle>::const_iterator it(CandHandleMC->begin()), itEnd(CandHandleMC->end());
	   it != itEnd;it++)
	{


	  //======
	  // 	  bool found( false );
	  // 	  for (int iParticle(0), iParticleEnd( pdgidList.size() ); iParticle<iParticleEnd; ++iParticle) 
	  // 	    {
	  // 	      //cout << "Particle pdgid " << it->pdgId() << " charge " << it->charge() << endl; 
	  // 	      if ( it->pdgId()==pdgidList[iParticle] )
	  // 		{
	  // 		  found = true;
	  // 		  break;
	  // 		}
	  // 	    }
	  // 	  if (!found) 
	  // 	    {
	  // 	      //cout << "Particle pdgid " << it->pdgId() << " status " << it->status() << endl; 
	  // 	      //cout << "Particle pdgid " << it->pdgId() << " charge " << it->charge() << endl;
	  // 	      pdgidList.push_back( it->pdgId() );
	  // 	    }
	  //======


	  GenPart.push_back(it->p4());
	}

      std::stable_sort(GenPart.begin(),GenPart.end(),GreaterPt());

      std::vector<math::XYZTLorentzVector>::const_iterator it(GenPart.begin()), itEnd(GenPart.end());
      for( int iMonteCarlo(0); it != itEnd; ++it, ++iMonteCarlo )
	{
	 
	  new((*MonteCarlo)[iMonteCarlo]) TLorentzVector(it->Px(), it->Py(), it->Pz(), it->E());
	
	}
    }

  } 

  
  // reco level analysis

  std::vector<math::XYZTLorentzVector> Tracks;

  std::vector<math::XYZPoint> AssVertices;

  std::vector<reco::CaloJet> RecoCaloJetContainer;

 std::vector<reco::TrackJet> TracksJetContainer;

  Tracks.clear();

  RecoCaloJetContainer.clear();

  TracksJetContainer.clear();

  //new   
  std::vector<math::XYZTLorentzVector> TracksT;

  TracksT.clear();

  Track->Clear();

  TracksJet->Clear();

  CalorimeterJet->Clear();


  if ( e.getByLabel( recoCaloJetCollName, RecoCaloJetsHandle ) )
    {
      if(RecoCaloJetsHandle->size())
	{
	  for(reco::CaloJetCollection::const_iterator it(RecoCaloJetsHandle->begin()), itEnd(RecoCaloJetsHandle->end());
	      it!=itEnd;++it)
	    {
	      RecoCaloJetContainer.push_back(*it);
	    }
	  std::stable_sort(RecoCaloJetContainer.begin(),RecoCaloJetContainer.end(),CaloJetSort());
	  
	  std::vector<reco::CaloJet>::const_iterator it(RecoCaloJetContainer.begin()), itEnd(RecoCaloJetContainer.end());
	  for( int iCalorimeterJet(0); it != itEnd; ++it, ++iCalorimeterJet)
	    {
	      new((*CalorimeterJet)[iCalorimeterJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	    }
	}
    }
    
  



if ( e.getByLabel( tracksJetCollName, TracksJetsHandle ) )
    {
      if(TracksJetsHandle->size())
	{
	  for(reco::TrackJetCollection::const_iterator it(TracksJetsHandle->begin()), itEnd(TracksJetsHandle->end());
	      it!=itEnd;++it)
	    {
	      TracksJetContainer.push_back(*it);

	      // 	      Jet::Constituents constituents( (*it).getJetConstituents() );
	      // 	      //cout << "get " << constituents.size() << " constituents" << endl;
	      // 	      for (int iJC(0); iJC<constituents.size(); ++iJC )
	      // 		{
	      // 		  //cout << "[" << iJC << "] constituent pT = " << constituents[iJC]->pt() << endl;
	      // 		  if (constituents[iJC]->et()<0.5)
	      // 		    {
	      // 		      cout << "ERROR!!! [" << iJC << "] constituent pT = " << constituents[iJC]->pt() << endl;
	      // 		    }
	      // 		}
	    }
	  std::stable_sort(TracksJetContainer.begin(),TracksJetContainer.end(),BasicJetSort());
	  
          int iJCall=0;
	  std::vector<reco::TrackJet>::const_iterator it(TracksJetContainer.begin()), itEnd(TracksJetContainer.end());
	  for(int iTracksJet(0); it != itEnd; ++it, ++iTracksJet)
	    {

	      reco::Jet::Constituents constituents( (*it).getJetConstituents() );
	      trackinjet_.tkn[iTracksJet]=constituents.size();
	      for (int iJC(0); iJC<(int)constituents.size(); ++iJC )
		{
		 
		  trackinjet_.tkp[iJCall]=constituents[iJC]->p();
		  trackinjet_.tkpt[iJCall]=constituents[iJC]->pt();
		  trackinjet_.tketa[iJCall]=constituents[iJC]->eta();
		  trackinjet_.tkphi[iJCall]=constituents[iJC]->phi();
		  //  trackinjet_.tknhit[iJCall]=constituents[iJC]->recHitsSize();
		  //  trackinjet_.tkchi2norm[iJCall]=constituents[iJC]->normalizedChi2();
		  //  trackinjet_.tkd0[iJCall]=constituents[iJC]->d0();
		  //  trackinjet_.tkd0Err[iJCall]=constituents[iJC]->d0Err();
		  //  trackinjet_.tkdz[iJCall]=constituents[iJC]->dz();
		  //  trackinjet_.tkdzErr[iJCall]=constituents[iJC]->dzErr();
		  iJCall++;
		}

	      new((*TracksJet)[iTracksJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	    }
	}
    }

 int iTracks=0;
  if ( e.getByLabel( tracksCollName , CandHandleRECO ) )
    {
      if(CandHandleRECO->size())
	{
	  
	  //for(CandidateCollection::const_iterator it(CandHandleRECO->begin()), itEnd(CandHandleRECO->end());
	  for(edm::View<reco::Candidate>::const_iterator it(CandHandleRECO->begin()), itEnd(CandHandleRECO->end());
	      it!=itEnd;++it)
	    {

    	     Tracks.push_back(it->p4());

	     
	     iTracks++; 
	    }
	   std::stable_sort(Tracks.begin(),Tracks.end(),GreaterPt());
	  
	    std::vector<math::XYZTLorentzVector>::const_iterator it( Tracks.begin()), itEnd(Tracks.end());
	     for(int iTracks(0); it != itEnd; ++it, ++iTracks)
	    {
	     new ((*Track)[iTracks]) TLorentzVector(it->Px(), it->Py(), it->Pz(), it->E());
	  
	     }
	}
    }
  
  store();
}

void AnalysisRootpleProducer::endJob()
{
  //   cout << "Printing list of PDG id's: " << endl;
  //   std::sort(pdgidList.begin(), pdgidList.end());
  //   for (int iParticle(0), iParticleEnd( pdgidList.size() ); iParticle<iParticleEnd; ++iParticle)
  //     {
  //       cout << pdgidList[iParticle] << endl;
  //     }
}

