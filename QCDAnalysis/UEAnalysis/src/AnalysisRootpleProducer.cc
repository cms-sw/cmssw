// Authors: F. Ambroglini, L. Fano', F. Bechtel
#include <QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducer.h>
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

using namespace edm;
using namespace std;
using namespace reco;

class GreaterPt{
public:
  bool operator()( const math::XYZTLorentzVector& a, const math::XYZTLorentzVector& b) {
    return a.pt() > b.pt();
  }
};

class GenJetSort{
public:
  bool operator()(const GenJet& a, const GenJet& b) {
    return a.pt() > b.pt();
  }
};

class BasicJetSort{
public:
  bool operator()(const BasicJet& a, const BasicJet& b) {
    return a.pt() > b.pt();
  }
};

class CaloJetSort{
public:
  bool operator()(const CaloJet& a, const CaloJet& b) {
    return a.pt() > b.pt();
  }
};
 
void AnalysisRootpleProducer::store(){

  AnalysisTree->Fill();

}

void AnalysisRootpleProducer::fillEventInfo(int e){
  EventKind = e;
}


AnalysisRootpleProducer::AnalysisRootpleProducer( const ParameterSet& pset )
{
  // flag to ignore gen-level analysis
  onlyRECO = pset.getParameter<bool>("OnlyRECO");

  // particle, track and jet collections
  mcEvent             = pset.getParameter<InputTag>( "MCEvent"                   );
  genJetCollName      = pset.getParameter<InputTag>( "GenJetCollectionName"      );
  chgJetCollName      = pset.getParameter<InputTag>( "ChgGenJetCollectionName"   );
  tracksJetCollName   = pset.getParameter<InputTag>( "TracksJetCollectionName"   );
  recoCaloJetCollName = pset.getParameter<InputTag>( "RecoCaloJetCollectionName" );
  chgGenPartCollName  = pset.getParameter<InputTag>( "ChgGenPartCollectionName"  );
  tracksCollName      = pset.getParameter<InputTag>( "TracksCollectionName"      );
  genEventScaleTag    = pset.getParameter<InputTag>( "genEventScale"             );

  //   cout << genJetCollName.label() << endl;
  //   cout << chgJetCollName.label() << endl;
  //   cout << tracksJetCollName.label() << endl;
  //   cout << recoCaloJetCollName.label() << endl;

  // trigger results
  triggerResultsTag = pset.getParameter<InputTag>("triggerResults");
  triggerEventTag   = pset.getParameter<InputTag>("triggerEvent"  );
  //   hltFilterTag      = pset.getParameter<InputTag>("hltFilter");
  //   triggerName       = pset.getParameter<InputTag>("triggerName");

  piG = acos(-1.);
  pdgidList.reserve(200);
}

void AnalysisRootpleProducer::beginJob( const EventSetup& )
{
 
  // use TFileService for output to root file
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
}

  
void AnalysisRootpleProducer::analyze( const Event& e, const EventSetup& )
{
  ///
  /// Pythia: genEventScaleTag = "genEventScale"
  /// Herwig: genEventScaleTag = "genEventKTValue"
  ///

  // if ( e.getByLabel( genEventScaleTag, genEventScaleHandle ) ) genEventScale = *genEventScaleHandle;

  Handle<GenEventInfoProduct> hEventInfo;
  e.getByLabel(genEventScaleTag , hEventInfo);
 if (hEventInfo->binningValues().size() > 0)
   { 
    genEventScale = hEventInfo->binningValues()[0];
   }  

// access trigger bits by TriggerEvent
  //   acceptedTriggers->Clear();
  //   unsigned int iAcceptedTriggers( 0 );
  //   if (e.getByLabel( triggerEventTag, triggerEvent ) )
  //     {
  //        // look at TriggerEvent 
  
  //       LogDebug("UEAnalysis") << "triggerEvent has " << triggerEvent.product()->sizeFilters() << " filters and "
  // 			     << "triggerEvent has " << triggerEvent.product()->sizeObjects() << " objects";
  
  //       LogDebug("UEAnalysis") << "size of object collection is " << triggerEvent.product()->getObjects().size() 
  // 			     << "usedProcessName() " << triggerEvent.product()->usedProcessName();
  
  //       for ( size_type index( 0 ) ; index < triggerEvent.product()->sizeFilters() ; ++index )
  // 	{
  // 	  LogDebug("UEAnalysis") << "filterLabel(size_type index) " << triggerEvent.product()->filterLabel(index);
  
  // 	  // save name of accepted trigger
  // 	  new((*acceptedTriggers)[iAcceptedTriggers]) TObjString( triggerEvent.product()->filterLabel(index).c_str() );
  // 	  ++iAcceptedTriggers;  
  // 	}
  //     }


  // access trigger bits by TriggerResults
  if (e.getByLabel( triggerResultsTag, triggerResults ) )
    {
      triggerNames.init( *(triggerResults.product()) );
      
      acceptedTriggers->Clear();
      unsigned int iAcceptedTriggers( 0 ); 
      if ( triggerResults.product()->wasrun() )
   	{
   	  LogDebug("UEAnalysis") << "at least one path out of " << triggerResults.product()->size() 
				 << " ran? " << triggerResults.product()->wasrun();
  
   	  if ( triggerResults.product()->accept() ) 
   	    {
   	      LogDebug("UEAnalysis") << "at least one path accepted? " << triggerResults.product()->accept() ;
  
   	      const unsigned int n_TriggerResults( triggerResults.product()->size() );
   	      for ( unsigned int itrig( 0 ); itrig < n_TriggerResults; ++itrig )
   		{
   		  LogDebug("UEAnalysis") << "path " << triggerNames.triggerName( itrig ) 
   					 << ", module index " << triggerResults.product()->index( itrig )
   					 << ", state (Ready = 0, Pass = 1, Fail = 2, Exception = 3) " << triggerResults.product()->state( itrig )
   					 << ", accept " << triggerResults.product()->accept( itrig );
  
     		  if ( triggerResults.product()->accept( itrig ) )
   		    {
   		      // save name of accepted trigger path
   		      new((*acceptedTriggers)[iAcceptedTriggers]) TObjString( (triggerNames.triggerName( itrig )).c_str() );
   		      ++iAcceptedTriggers;
   		    }
   		}
   	    }
   	}
    }

  // gen level analysis
  // skipped, if onlyRECO flag set to true
  
  if(!onlyRECO){
    
    e.getByLabel( mcEvent           , EvtHandle        );
    e.getByLabel( chgGenPartCollName, CandHandleMC     );
    e.getByLabel( chgJetCollName    , ChgGenJetsHandle );
    e.getByLabel( genJetCollName    , GenJetsHandle    );

    const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
    
    EventKind = Evt->signal_process_id();

    std::vector<math::XYZTLorentzVector> GenPart;
    std::vector<GenJet> ChgGenJetContainer;
    std::vector<GenJet> GenJetContainer;
    
    GenPart.clear();
    ChgGenJetContainer.clear();
    GenJetContainer.clear();
    MonteCarlo->Clear();
    InclusiveJet->Clear();
    ChargedJet->Clear();

    // jets from charged particles at hadron level
    if (ChgGenJetsHandle->size()){

      for ( GenJetCollection::const_iterator it(ChgGenJetsHandle->begin()), itEnd(ChgGenJetsHandle->end());
	    it!=itEnd; ++it)
	{
	  ChgGenJetContainer.push_back(*it);
	}

      std::stable_sort(ChgGenJetContainer.begin(),ChgGenJetContainer.end(),GenJetSort());

      std::vector<GenJet>::const_iterator it(ChgGenJetContainer.begin()), itEnd(ChgGenJetContainer.end());
      for ( int iChargedJet(0); it != itEnd; ++it, ++iChargedJet)
	{
	  new((*ChargedJet)[iChargedJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	}
    }


    // GenJets
    if (GenJetsHandle->size()){

      for ( GenJetCollection::const_iterator it(GenJetsHandle->begin()), itEnd(GenJetsHandle->end());
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

      std::vector<GenJet>::const_iterator it(GenJetContainer.begin()), itEnd(GenJetContainer.end());
      for ( int iInclusiveJet(0); it != itEnd; ++it, ++iInclusiveJet)
	{
	  new((*InclusiveJet)[iInclusiveJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	}
    }


    // hadron level particles
    if (CandHandleMC->size()){
      
      for (vector<GenParticle>::const_iterator it(CandHandleMC->begin()), itEnd(CandHandleMC->end());
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
  std::vector<CaloJet> RecoCaloJetContainer;
  std::vector<BasicJet> TracksJetContainer;

  Tracks.clear();
  RecoCaloJetContainer.clear();
  TracksJetContainer.clear();
  
  Track->Clear();
  TracksJet->Clear();
  CalorimeterJet->Clear();

  if ( e.getByLabel( recoCaloJetCollName, RecoCaloJetsHandle ) )
    {
      if(RecoCaloJetsHandle->size())
	{
	  for(CaloJetCollection::const_iterator it(RecoCaloJetsHandle->begin()), itEnd(RecoCaloJetsHandle->end());
	      it!=itEnd;++it)
	    {
	      RecoCaloJetContainer.push_back(*it);
	    }
	  std::stable_sort(RecoCaloJetContainer.begin(),RecoCaloJetContainer.end(),CaloJetSort());
	  
	  std::vector<CaloJet>::const_iterator it(RecoCaloJetContainer.begin()), itEnd(RecoCaloJetContainer.end());
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
	  for(BasicJetCollection::const_iterator it(TracksJetsHandle->begin()), itEnd(TracksJetsHandle->end());
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
	  
	  std::vector<BasicJet>::const_iterator it(TracksJetContainer.begin()), itEnd(TracksJetContainer.end());
	  for(int iTracksJet(0); it != itEnd; ++it, ++iTracksJet)
	    {
	      new((*TracksJet)[iTracksJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	    }
	}
    }
  

  if ( e.getByLabel( tracksCollName , CandHandleRECO ) )
    {
      if(CandHandleRECO->size())
	{
	  //for(CandidateCollection::const_iterator it(CandHandleRECO->begin()), itEnd(CandHandleRECO->end());
	  for(edm::View<reco::Candidate>::const_iterator it(CandHandleRECO->begin()), itEnd(CandHandleRECO->end());
	      it!=itEnd;++it)
	    {
	      Tracks.push_back(it->p4());
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

