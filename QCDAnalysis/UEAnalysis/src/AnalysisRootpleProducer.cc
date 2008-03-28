// Authors: F. Ambroglini, L. Fano'
#include <iostream>

#include "QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducer.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
 
 
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

  NumberMCParticles=0;
  NumberTracks=0;
  NumberInclusiveJet=0;
  NumberChargedJet=0;
  NumberTracksJet=0;
  NumberCaloJet=0;
}

void AnalysisRootpleProducer::fillEventInfo(int e){
  EventKind = e;
}

void AnalysisRootpleProducer::fillMCParticles(float p, float pt, float eta, float phi){
  MomentumMC[NumberMCParticles]=p;
  TransverseMomentumMC[NumberMCParticles]=pt;
  EtaMC[NumberMCParticles]=eta;
  PhiMC[NumberMCParticles]=phi;
  NumberMCParticles++;
}

void AnalysisRootpleProducer::fillTracks(float p, float pt, float eta, float phi){
  MomentumTK[NumberTracks]=p;
  TransverseMomentumTK[NumberTracks]=pt;
  EtaTK[NumberTracks]=eta;
  PhiTK[NumberTracks]=phi;
  NumberTracks++;
}

void AnalysisRootpleProducer::fillInclusiveJet(float p, float pt, float eta, float phi){
  MomentumIJ[NumberInclusiveJet]=p;
  TransverseMomentumIJ[NumberInclusiveJet]=pt;
  EtaIJ[NumberInclusiveJet]=eta;
  PhiIJ[NumberInclusiveJet]=phi;
  NumberInclusiveJet++;
}

void AnalysisRootpleProducer::fillChargedJet(float p, float pt, float eta, float phi){
  MomentumCJ[NumberChargedJet]=p;
  TransverseMomentumCJ[NumberChargedJet]=pt;
  EtaCJ[NumberChargedJet]=eta;
  PhiCJ[NumberChargedJet]=phi;
  NumberChargedJet++;
}

void AnalysisRootpleProducer::fillTracksJet(float p, float pt, float eta, float phi){
  MomentumTJ[NumberTracksJet]=p;
  TransverseMomentumTJ[NumberTracksJet]=pt;
  EtaTJ[NumberTracksJet]=eta;
  PhiTJ[NumberTracksJet]=phi;
  NumberTracksJet++;
}

void AnalysisRootpleProducer::fillCaloJet(float p, float pt, float eta, float phi){
  MomentumEHJ[NumberCaloJet]=p;
  TransverseMomentumEHJ[NumberCaloJet]=pt;
  EtaEHJ[NumberCaloJet]=eta;
  PhiEHJ[NumberCaloJet]=phi;
  NumberCaloJet++;
}

AnalysisRootpleProducer::AnalysisRootpleProducer( const ParameterSet& pset )
  : onlyRECO( pset.getUntrackedParameter<bool>("OnlyRECO",false)),
    mcEvent( pset.getUntrackedParameter<string>("MCEvent",std::string(""))),
    genJetCollName( pset.getUntrackedParameter<string>("GenJetCollectionName",std::string(""))),
    chgJetCollName( pset.getUntrackedParameter<string>("ChgGenJetCollectionName",std::string(""))),
    tracksJetCollName( pset.getUntrackedParameter<string>("TracksJetCollectionName",std::string(""))),
    recoCaloJetCollName( pset.getUntrackedParameter<string>("RecoCaloJetCollectionName",std::string(""))),
    chgGenPartCollName( pset.getUntrackedParameter<string>("ChgGenPartCollectionName",std::string(""))),
    tracksCollName( pset.getUntrackedParameter<string>("TracksCollectionName",std::string("")))
{
  piG = acos(-1.);
  NumberMCParticles=0;
  NumberTracks=0;
  NumberInclusiveJet=0;
  NumberChargedJet=0;
  NumberTracksJet=0;
  NumberCaloJet=0;
}

void AnalysisRootpleProducer::beginJob( const EventSetup& )
{
 
  // use TFileService for output to root file
  AnalysisTree = fs->make<TTree>("AnalysisTree","MBUE Analysis Tree ");

//   hFile = new TFile ( fOutputFileName.c_str(), "RECREATE" );
//   AnalysisTree = new TTree("AnalysisTree","MBUE Analysis Tree ");
  
  AnalysisTree->Branch("EventKind",&EventKind,"EventKind/I");

  // store p, pt, eta, phi for particles and jets

  // GenParticles at hadron level  
  AnalysisTree->Branch("NumberMCParticles",&NumberMCParticles,"NumberMCParticles/I");
  AnalysisTree->Branch("MomentumMC",MomentumMC,"MomentumMC[NumberMCParticles]/F");
  AnalysisTree->Branch("TransverseMomentumMC",TransverseMomentumMC,"TransverseMomentumMC[NumberMCParticles]/F");
  AnalysisTree->Branch("EtaMC",EtaMC,"EtaMC[NumberMCParticles]/F");
  AnalysisTree->Branch("PhiMC",PhiMC,"PhiMC[NumberMCParticles]/F");
  
  // tracks
  AnalysisTree->Branch("NumberTracks",&NumberTracks,"NumberTracks/I");
  AnalysisTree->Branch("MomentumTK",MomentumTK,"MomentumTK[NumberTracks]/F");
  AnalysisTree->Branch("TrasverseMomentumTK",TransverseMomentumTK,"TransverseMomentumTK[NumberTracks]/F");
  AnalysisTree->Branch("EtaTK",EtaTK,"EtaTK[NumberTracks]/F");
  AnalysisTree->Branch("PhiTK",PhiTK,"PhiTK[NumberTracks]/F");
  
  // GenJets
  AnalysisTree->Branch("NumberInclusiveJet",&NumberInclusiveJet,"NumberInclusiveJet/I");
  AnalysisTree->Branch("MomentumIJ",MomentumIJ,"MomentumIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("TrasverseMomentumIJ",TransverseMomentumIJ,"TransverseMomentumIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("EtaIJ",EtaIJ,"EtaIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("PhiIJ",PhiIJ,"PhiIJ[NumberInclusiveJet]/F");
  
  // jets from charged GenParticles
  AnalysisTree->Branch("NumberChargedJet",&NumberChargedJet,"NumberChargedJet/I");
  AnalysisTree->Branch("MomentumCJ",MomentumCJ,"MomentumCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("TrasverseMomentumCJ",TransverseMomentumCJ,"TransverseMomentumCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("EtaCJ",EtaCJ,"EtaCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("PhiCJ",PhiCJ,"PhiCJ[NumberChargedJet]/F");
  
  // jets from tracks
  AnalysisTree->Branch("NumberTracksJet",&NumberTracksJet,"NumberTracksJet/I");
  AnalysisTree->Branch("MomentumTJ",MomentumTJ,"MomentumTJ[NumberTracksJet]/F");
  AnalysisTree->Branch("TrasverseMomentumTJ",TransverseMomentumTJ,"TransverseMomentumTJ[NumberTracksJet]/F");
  AnalysisTree->Branch("EtaTJ",EtaTJ,"EtaTJ[NumberTracksJet]/F");
  AnalysisTree->Branch("PhiTJ",PhiTJ,"PhiTJ[NumberTracksJet]/F");
  
  // jets from calorimeter towers
  AnalysisTree->Branch("NumberCaloJet",&NumberCaloJet,"NumberCaloJet/I");
  AnalysisTree->Branch("MomentumEHJ",MomentumEHJ,"MomentumEHJ[NumberCaloJet]/F");
  AnalysisTree->Branch("TrasverseMomentumEHJ",TransverseMomentumEHJ,"TransverseMomentumEHJ[NumberCaloJet]/F");
  AnalysisTree->Branch("EtaEHJ",EtaEHJ,"EtaEHJ[NumberCaloJet]/F");
  AnalysisTree->Branch("PhiEHJ",PhiEHJ,"PhiEHJ[NumberCaloJet]/F");
  

  // alternative storage method:
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
}

  
void AnalysisRootpleProducer::analyze( const Event& e, const EventSetup& )
{
  
  // gen level analysis
  // skipped, if onlyRECO flag set to true

  if(!onlyRECO){
    
    Handle< HepMCProduct        > EvtHandle ;
    Handle< CandidateCollection > CandHandleMC ;
    Handle< GenJetCollection    > GenJetsHandle ;
    Handle< GenJetCollection    > ChgGenJetsHandle ;
    
    e.getByLabel( mcEvent.c_str()           , EvtHandle        );
    e.getByLabel( chgGenPartCollName.c_str(), CandHandleMC     );
    e.getByLabel( chgJetCollName.c_str()    , ChgGenJetsHandle );
    e.getByLabel( genJetCollName.c_str()    , GenJetsHandle    );
    
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
	  fillChargedJet(it->p(),it->pt(),it->eta(),it->phi());
	  new((*ChargedJet)[iChargedJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	}
    }


    // GenJets
    if (GenJetsHandle->size()){

      for ( GenJetCollection::const_iterator it(GenJetsHandle->begin()), itEnd(GenJetsHandle->end());
	    it!=itEnd; ++it )
	{
	  GenJetContainer.push_back(*it);
	}

      std::stable_sort(GenJetContainer.begin(),GenJetContainer.end(),GenJetSort());

      std::vector<GenJet>::const_iterator it(GenJetContainer.begin()), itEnd(GenJetContainer.end());
      for ( int iInclusiveJet(0); it != itEnd; ++it, ++iInclusiveJet)
	{
	  fillInclusiveJet(it->p(),it->pt(),it->eta(),it->phi());
	  new((*InclusiveJet)[iInclusiveJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	}
    }


    // hadron level particles
    if (CandHandleMC->size()){

      for (CandidateCollection::const_iterator it(CandHandleMC->begin()), itEnd(CandHandleMC->end());
	   it != itEnd;it++)
	{
	  GenPart.push_back(it->p4());
	}

      std::stable_sort(GenPart.begin(),GenPart.end(),GreaterPt());

      std::vector<math::XYZTLorentzVector>::const_iterator it(GenPart.begin()), itEnd(GenPart.end());
      for( int iMonteCarlo(0); it != itEnd; ++it, ++iMonteCarlo )
	{
	  fillMCParticles(it->P(),it->Pt(),it->Eta(),it->Phi());
	  new((*MonteCarlo)[iMonteCarlo]) TLorentzVector(it->Px(), it->Py(), it->Pz(), it->E());
	}
    }

  } 

  
  // reco level analysis

  Handle< CandidateCollection > CandHandleRECO ;
  Handle< BasicJetCollection  > TracksJetsHandle ;
  Handle< CaloJetCollection   > RecoCaloJetsHandle ;
  
  e.getByLabel( tracksCollName.c_str()     , CandHandleRECO     );
  e.getByLabel( recoCaloJetCollName.c_str(), RecoCaloJetsHandle );
  e.getByLabel( tracksJetCollName.c_str()  , TracksJetsHandle   );
  
  std::vector<math::XYZTLorentzVector> Tracks;
  std::vector<BasicJet> TracksJetContainer;
  std::vector<CaloJet> RecoCaloJetContainer;
  
  Tracks.clear();
  TracksJetContainer.clear();
  RecoCaloJetContainer.clear();
  
  Track->Clear();
  TracksJet->Clear();
  CalorimeterJet->Clear();

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
	fillCaloJet(it->p(),it->pt(),it->eta(),it->phi());
	new((*CalorimeterJet)[iCalorimeterJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
      }
    }
    
  if(TracksJetsHandle->size())
    {
      for(BasicJetCollection::const_iterator it(TracksJetsHandle->begin()), itEnd(TracksJetsHandle->end());
	  it!=itEnd;++it)
	{
	  TracksJetContainer.push_back(*it);
	}
      std::stable_sort(TracksJetContainer.begin(),TracksJetContainer.end(),BasicJetSort());
    
      std::vector<BasicJet>::const_iterator it(TracksJetContainer.begin()), itEnd(TracksJetContainer.end());
      for(int iTracksJet(0); it != itEnd; ++it, ++iTracksJet)
	{
	  fillTracksJet(it->p(),it->pt(),it->eta(),it->phi());
	  new((*TracksJet)[iTracksJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
	}
    }
  
  if(CandHandleRECO->size())
    {
      for(CandidateCollection::const_iterator it(CandHandleRECO->begin()), itEnd(CandHandleRECO->end());
	  it!=itEnd;++it)
	{
	  Tracks.push_back(it->p4());
	}
      std::stable_sort(Tracks.begin(),Tracks.end(),GreaterPt());
    
      std::vector<math::XYZTLorentzVector>::const_iterator it( Tracks.begin()), itEnd(Tracks.end());
      for(int iTracks(0); it != itEnd; ++it, ++iTracks)
	{
	  fillTracks(it->P(),it->Pt(),it->Eta(),it->Phi());
	  new ((*Track)[iTracks]) TLorentzVector(it->Px(), it->Py(), it->Pz(), it->E());
	}
    }
  
  store();
}

void AnalysisRootpleProducer::endJob()
{
}

