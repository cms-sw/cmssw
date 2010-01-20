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

void AnalysisRootpleProducer::beginJob()
{
 
  // use TFileService for output to root file
  AnalysisTree = fs->make<TTree>("AnalysisTree","MBUE Analysis Tree ");

  AnalysisTree->Branch("EventKind",&EventKind,"EventKind/I");

  // save TClonesArrays of TLorentzVectors
  // i.e. store 4-vectors of particles and jets
  
 
  MonteCarlo = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("MonteCarlo", "TClonesArray", &MonteCarlo, 128000, 0);

  MonteCarlo2 = new TClonesArray("TVector", 10000);
  AnalysisTree->Branch("MonteCarlo2", "TClonesArray", &MonteCarlo2,128000, 0);

  Track = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("Track", "TClonesArray", &Track, 128000, 0);

  AssVertex = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("AssVertex", "TClonesArray", &AssVertex, 128000, 0);

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

  //AnalysisTree->Branch("npv",&m_npv,"npv/I");
  //AnalysisTree->Branch("pvx",m_pvx, "pvx[npv]/D");
  //AnalysisTree->Branch("pvy",m_pvy, "pvy[npv]/D");
  //AnalysisTree->Branch("pvz",m_pvz, "pvz[npv]/D");
  //AnalysisTree->Branch("pvxErr",m_pvxErr, "pvxErr[npv]/D");
  //AnalysisTree->Branch("pvyErr",m_pvyErr, "pvyErr[npv]/D");
  //AnalysisTree->Branch("pvzErr",m_pvzErr, "pvzErr[npv]/D");
  //  AnalysisTree->Branch("pvntk", m_pvntk, "pvntk[npv]/I");
  
  //AnalysisTree->Branch("pvtkp",m_pvtkp,"pvtkp[npv]/D");
  //AnalysisTree->Branch("pvtkpt",m_pvtkpt,"pvtkpt[5000]/D");
  //AnalysisTree->Branch("pvtketa",m_pvtketa,"pvtketa[5000]/D");
  //AnalysisTree->Branch("pvtkphi",m_pvtkphi,"pvtkphi[5000]/D");
  //AnalysisTree->Branch("pvtkchi2norm",m_pvtkchi2norm,"pvtkchi2norm[5000]/D");
  //AnalysisTree->Branch("pvtknhit",m_pvtknhit,"pvtknhit[5000]/D");
  //AnalysisTree->Branch("pvtkd0",m_pvtkd0,"pvtkd0[5000]/D");
  //AnalysisTree->Branch("pvtkd0Err",m_pvtkd0Err,"pvtkd0Err[5000]/D");
  //AnalysisTree->Branch("pvtkdz",m_pvtkdz,"pvtkdz[5000]/D");
  //AnalysisTree->Branch("pvtkdzErr",m_pvtkdzErr,"pvtkdzErr[5000]/D");

  //AnalysisTree->Branch("ntk",&m_ntk,"ntk/I");
  //AnalysisTree->Branch("tkpt",m_tkpt,"tkpt[ntk]/D");
  //AnalysisTree->Branch("tketa",m_tketa,"tketa[ntk]/D");
  //AnalysisTree->Branch("tkphi",m_tkphi,"tkphi[ntk]/D");
  //AnalysisTree->Branch("tkchi2norm",m_tkchi2norm,"tkchi2norm[ntk]/D");
  //AnalysisTree->Branch("tknhit",m_tknhit,"tknhit[ntk]/D");
  //AnalysisTree->Branch("tkd0",m_tkd0,"tkd0[ntk]/D");
  //AnalysisTree->Branch("tkd0Err",m_tkd0Err,"tkd0Err[ntk]/D");
  //AnalysisTree->Branch("tkdz",m_tkdz,"tkdz[ntk]/D");
  //AnalysisTree->Branch("tkdzErr",m_tkdzErr,"tkdzErr[ntk]/D");

}

  
void AnalysisRootpleProducer::analyze( const Event& e, const EventSetup& )
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
  Handle<GenEventInfoProduct> hEventInfo;
  e.getByLabel(genEventScaleTag , hEventInfo);
 if (hEventInfo->binningValues().size() > 0)
   { 
    genEventScale = hEventInfo->binningValues()[0];
   }  
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


    //Primary Vertex
 //primary vertex extraction --------------------------------------------------------------------------

  int ipv = 0;
  edm::Handle<reco::VertexCollection> primaryVertexHandle;
  e.getByLabel("offlinePrimaryVertices",primaryVertexHandle);
  /* 
 if(primaryVertexHandle->size()>0){

for(reco::VertexCollection::const_iterator it = primaryVertexHandle->begin(), ed = primaryVertexHandle->end();
	 it != ed; ++it) {
  reco::Vertex pv;
  pv = (*it);
  m_pvx[ipv] = pv.x();
  m_pvy[ipv] = pv.y();
  m_pvz[ipv] = pv.z();
  
  m_pvxErr[ipv] = pv.xError();
  m_pvyErr[ipv] = pv.yError();
  m_pvzErr[ipv] = pv.zError();
  
  // int ipvtk=0;
  // for(reco::Vertex::trackRef_iterator pvt = pv.tracks_begin(); pvt!= pv.tracks_end(); pvt++){
  // const reco::Track & track = *pvt->get();
  //  
  // m_pvtkpt[ipv+ipvtk]=track.pt();
  //  m_pvtkp[ipv+ipvtk]=track.p();
  //  m_pvtketa[ipv+ipvtk]=track.eta();
  //  m_pvtkphi[ipv+ipvtk]=track.phi();
  //  m_pvtkchi2norm[ipv+ipvtk]=track.normalizedChi2();
  //  m_pvtkd0[ipv+ipvtk]=track.d0();
  //  m_pvtkd0Err[ipv+ipvtk]=track.d0Error();
  //  m_pvtkdz[ipv+ipvtk]=track.dz();
  //  m_pvtkdzErr[ipv+ipvtk]=track.dzError();
  //  m_pvtknhit[ipv+ipvtk]=track.recHitsSize();
       
  // ipvtk++;
  // }
  //   m_pvntk[ipv] = ipvtk;
   ipv++;
}
  }
  m_npv = ipv;
*/



  ///-------------------------------
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
  std::vector<math::XYZPoint> AssVertices; 
  std::vector<CaloJet> RecoCaloJetContainer;
  std::vector<BasicJet> TracksJetContainer;

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

// edm::Handle< reco::TrackCollection  > trackColl;
// e.getByLabel(tracksCollName,trackColl);


// for (reco::TrackCollection::const_iterator it = trackColl->begin();
//                                           it != trackColl->end();
//                                           it++){
//   m_tkpt[m_ntk]=it->pt();
//   m_tkp[m_ntk]=it->p();
//   m_tketa[m_ntk]=it->eta();
//   m_tkphi[m_ntk]=it->phi();
//   m_tkchi2norm[m_ntk]=it->normalizedChi2();
//   m_tkd0[m_ntk]=it->d0();
//   m_tkd0Err[m_ntk]=it->d0Error();
//   m_tkdz[m_ntk]=it->dz();
//   m_tkdzErr[m_ntk]=it->dzError();
    
//   m_ntk++;    
//   }


 int iTracks=0;
  if ( e.getByLabel( tracksCollName , CandHandleRECO ) )
    {
      if(CandHandleRECO->size())
	{
	  
	  //for(CandidateCollection::const_iterator it(CandHandleRECO->begin()), itEnd(CandHandleRECO->end());
	  for(edm::View<reco::Candidate>::const_iterator it(CandHandleRECO->begin()), itEnd(CandHandleRECO->end());
	      it!=itEnd;++it)
	    {
	      new ((*Track)[iTracks]) TLorentzVector(it->p4().Px(), it->p4().Py(), it->p4().Pz(), it->p4().E());
	      //Tracks.push_back(it->p4());
	      // AssVertices.push_back(it->vertex());	
	     new ((*AssVertex)[iTracks]) TLorentzVector(it->vertex().x(),it->vertex().y(),it->vertex().z(),0); 
	     iTracks++; 
	    }
	  // std::stable_sort(Tracks.begin(),Tracks.end(),GreaterPt());
	  
	  //  std::vector<math::XYZTLorentzVector>::const_iterator it( Tracks.begin()), itEnd(Tracks.end());
	  //   for(int iTracks(0); it != itEnd; ++it, ++iTracks)
	  //  {
	  //   new ((*Track)[iTracks]) TLorentzVector(it->Px(), it->Py(), it->Pz(), it->E());
	  
	  //   }
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

