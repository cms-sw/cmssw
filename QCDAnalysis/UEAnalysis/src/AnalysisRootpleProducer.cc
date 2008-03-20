// Authors: F. Ambroglini, L. Fano'
#include <iostream>

#include "QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducer.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"
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
  : fOutputFileName( pset.getUntrackedParameter<string>("HistOutFile",std::string("TestHiggsMass.root")) ),
    onlyRECO( pset.getUntrackedParameter<bool>("OnlyRECO",false)),
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
 
  hFile = new TFile ( fOutputFileName.c_str(), "RECREATE" );
  AnalysisTree = new TTree("AnalysisTree","MBUE Analysis Tree ");
  
  AnalysisTree->Branch("EventKind",&EventKind,"EventKind/I");
  
  AnalysisTree->Branch("NumberMCParticles",&NumberMCParticles,"NumberMCParticles/I");
  AnalysisTree->Branch("MomentumMC",MomentumMC,"MomentumMC[NumberMCParticles]/F");
  AnalysisTree->Branch("TransverseMomentumMC",TransverseMomentumMC,"TransverseMomentumMC[NumberMCParticles]/F");
  AnalysisTree->Branch("EtaMC",EtaMC,"EtaMC[NumberMCParticles]/F");
  AnalysisTree->Branch("PhiMC",PhiMC,"PhiMC[NumberMCParticles]/F");
  
  AnalysisTree->Branch("NumberTracks",&NumberTracks,"NumberTracks/I");
  AnalysisTree->Branch("MomentumTK",MomentumTK,"MomentumTK[NumberTracks]/F");
  AnalysisTree->Branch("TrasverseMomentumTK",TransverseMomentumTK,"TransverseMomentumTK[NumberTracks]/F");
  AnalysisTree->Branch("EtaTK",EtaTK,"EtaTK[NumberTracks]/F");
  AnalysisTree->Branch("PhiTK",PhiTK,"PhiTK[NumberTracks]/F");
  
  AnalysisTree->Branch("NumberInclusiveJet",&NumberInclusiveJet,"NumberInclusiveJet/I");
  AnalysisTree->Branch("MomentumIJ",MomentumIJ,"MomentumIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("TrasverseMomentumIJ",TransverseMomentumIJ,"TransverseMomentumIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("EtaIJ",EtaIJ,"EtaIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("PhiIJ",PhiIJ,"PhiIJ[NumberInclusiveJet]/F");
  
  AnalysisTree->Branch("NumberChargedJet",&NumberChargedJet,"NumberChargedJet/I");
  AnalysisTree->Branch("MomentumCJ",MomentumCJ,"MomentumCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("TrasverseMomentumCJ",TransverseMomentumCJ,"TransverseMomentumCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("EtaCJ",EtaCJ,"EtaCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("PhiCJ",PhiCJ,"PhiCJ[NumberChargedJet]/F");
  
  AnalysisTree->Branch("NumberTracksJet",&NumberTracksJet,"NumberTracksJet/I");
  AnalysisTree->Branch("MomentumTJ",MomentumTJ,"MomentumTJ[NumberTracksJet]/F");
  AnalysisTree->Branch("TrasverseMomentumTJ",TransverseMomentumTJ,"TransverseMomentumTJ[NumberTracksJet]/F");
  AnalysisTree->Branch("EtaTJ",EtaTJ,"EtaTJ[NumberTracksJet]/F");
  AnalysisTree->Branch("PhiTJ",PhiTJ,"PhiTJ[NumberTracksJet]/F");
  
  AnalysisTree->Branch("NumberCaloJet",&NumberCaloJet,"NumberCaloJet/I");
  AnalysisTree->Branch("MomentumEHJ",MomentumEHJ,"MomentumEHJ[NumberCaloJet]/F");
  AnalysisTree->Branch("TrasverseMomentumEHJ",TransverseMomentumEHJ,"TransverseMomentumEHJ[NumberCaloJet]/F");
  AnalysisTree->Branch("EtaEHJ",EtaEHJ,"EtaEHJ[NumberCaloJet]/F");
  AnalysisTree->Branch("PhiEHJ",PhiEHJ,"PhiEHJ[NumberCaloJet]/F");
  
}

  
void AnalysisRootpleProducer::analyze( const Event& e, const EventSetup& )
{
  
  if(!onlyRECO){
    
    Handle< HepMCProduct > EvtHandle ;
    
    e.getByLabel( mcEvent.c_str(), EvtHandle ) ;
    
    const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
    
    EventKind = Evt->signal_process_id();
    
    Handle< CandidateCollection > CandHandleMC ;
    
    Handle< GenJetCollection > GenJetsHandle ;
    Handle< GenJetCollection > ChgGenJetsHandle ;
    
    e.getByLabel( chgGenPartCollName.c_str(), CandHandleMC );
    
    e.getByLabel(chgJetCollName.c_str(), ChgGenJetsHandle );
    e.getByLabel(genJetCollName.c_str(), GenJetsHandle );
    
    std::vector<math::XYZTLorentzVector> GenPart;
    
    std::vector<GenJet> ChgGenJetContainer;
    std::vector<GenJet> GenJetContainer;
    
    GenPart.clear();
    
    ChgGenJetContainer.clear();
    GenJetContainer.clear();
    
    if(ChgGenJetsHandle->size()){
      for(GenJetCollection::const_iterator it=ChgGenJetsHandle->begin();it!=ChgGenJetsHandle->end();it++)
	ChgGenJetContainer.push_back(*it);
      std::stable_sort(ChgGenJetContainer.begin(),ChgGenJetContainer.end(),GenJetSort());
      for(std::vector<GenJet>::const_iterator it = ChgGenJetContainer.begin(); it != ChgGenJetContainer.end(); it++)
	fillChargedJet(it->p(),it->pt(),it->eta(),it->phi());
    }
    
    if(GenJetsHandle->size()){
      for(GenJetCollection::const_iterator it=GenJetsHandle->begin();it!=GenJetsHandle->end();it++)
	GenJetContainer.push_back(*it);
      std::stable_sort(GenJetContainer.begin(),GenJetContainer.end(),GenJetSort());
      for(std::vector<GenJet>::const_iterator it = GenJetContainer.begin(); it != GenJetContainer.end(); it++)
	fillInclusiveJet(it->p(),it->pt(),it->eta(),it->phi());
  }
    
    if(CandHandleMC->size()){
      for(CandidateCollection::const_iterator it = CandHandleMC->begin();it!=CandHandleMC->end();it++){
	GenPart.push_back(it->p4());
      }
      std::stable_sort(GenPart.begin(),GenPart.end(),GreaterPt());
      for(std::vector<math::XYZTLorentzVector>::const_iterator it = GenPart.begin(); it != GenPart.end(); it++)
	fillMCParticles(it->P(),it->Pt(),it->Eta(),it->Phi());
    }
    
  } 
  
  Handle< CandidateCollection > CandHandleRECO ;
  
  Handle< BasicJetCollection > TracksJetsHandle ;
  Handle< CaloJetCollection > RecoCaloJetsHandle ;
  
  e.getByLabel( tracksCollName.c_str(), CandHandleRECO );
  
  e.getByLabel(recoCaloJetCollName.c_str(), RecoCaloJetsHandle );
  e.getByLabel(tracksJetCollName.c_str(), TracksJetsHandle );
  
  std::vector<math::XYZTLorentzVector> Tracks;
  std::vector<BasicJet> TracksJetContainer;
  std::vector<CaloJet> RecoCaloJetContainer;
  
  Tracks.clear();
  TracksJetContainer.clear();
  RecoCaloJetContainer.clear();
  
  if(RecoCaloJetsHandle->size()){
    for(CaloJetCollection::const_iterator it=RecoCaloJetsHandle->begin();it!=RecoCaloJetsHandle->end();it++)
      RecoCaloJetContainer.push_back(*it);
    std::stable_sort(RecoCaloJetContainer.begin(),RecoCaloJetContainer.end(),CaloJetSort());
    for(std::vector<CaloJet>::const_iterator it = RecoCaloJetContainer.begin(); it != RecoCaloJetContainer.end(); it++)
      fillCaloJet(it->p(),it->pt(),it->eta(),it->phi());
  }
    
  if(TracksJetsHandle->size()){
    for(BasicJetCollection::const_iterator it=TracksJetsHandle->begin();it!=TracksJetsHandle->end();it++)
	TracksJetContainer.push_back(*it);
    std::stable_sort(TracksJetContainer.begin(),TracksJetContainer.end(),BasicJetSort());
    for(std::vector<BasicJet>::const_iterator it = TracksJetContainer.begin(); it != TracksJetContainer.end(); it++)
      fillTracksJet(it->p(),it->pt(),it->eta(),it->phi());
  }
  
  if(CandHandleRECO->size()){
    for(CandidateCollection::const_iterator it = CandHandleRECO->begin();it!=CandHandleRECO->end();it++){
      Tracks.push_back(it->p4());
    }
    std::stable_sort(Tracks.begin(),Tracks.end(),GreaterPt());
    for(std::vector<math::XYZTLorentzVector>::const_iterator it = Tracks.begin(); it != Tracks.end(); it++)
      fillTracks(it->P(),it->Pt(),it->Eta(),it->Phi());
  }
  
  store();
}

void AnalysisRootpleProducer::endJob(){
  hFile->cd();
  AnalysisTree->Write();
  hFile->Close();
}

