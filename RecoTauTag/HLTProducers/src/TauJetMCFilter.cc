// Original Author: S. Gennai

// Modified by Author: K. Petridis S.Greder

#include "RecoTauTag/HLTProducers/interface/TauJetMCFilter.h"

//#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
//#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "Math/GenVector/VectorUtil.h"

#include "TLorentzVector.h"
//
// class decleration
//
using namespace edm;
using namespace HepMC;
using namespace std;

TauJetMCFilter::TauJetMCFilter(const edm::ParameterSet& iConfig)
{
  genParticles = iConfig.getParameter<InputTag>("GenParticles");
  mEtaMax = iConfig.getParameter<double>("EtaTauMax");
  mEtaMin = iConfig.getParameter<double>("EtaTauMin");
  mEtTau = iConfig.getParameter<double>("EtTau");
  mEtaElecMax = iConfig.getParameter<double>("EtaElecMax");
  mPtElec = iConfig.getParameter<double>("PtElec");
  mEtaMuonMax = iConfig.getParameter<double>("EtaMuonMax");
  mPtMuon = iConfig.getParameter<double>("PtMuon");
  mincludeList= iConfig.getParameter<vstring>( "includeList" );
  //tauParticles = iConfig.getParameter<InputTag>("TauParticles");
  
}

TauJetMCFilter::~TauJetMCFilter(){ }

HepMC::GenParticle * TauJetMCFilter::findParticle(const GenPartVect genPartVect,const int requested_id)
{
  for (GenPartVectIt p = genPartVect.begin(); p != genPartVect.end(); p++)
    {
      if (requested_id == (*p)->pdg_id()) return *p;
    }
  return 0;
}

bool TauJetMCFilter::filter(edm::Event& iEvent, const edm::EventSetup& iES)
{
  //
  //  Handle< L1JetParticleCollection > tauColl ; 
  //  iEvent.getByLabel( tauParticles, tauColl );
  //  const L1JetParticleCollection & myL1Tau  = *(tauColl.product());
  //
  
  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(genParticles, evt);
  
  HepMC::GenEvent * generated_event = new HepMC::GenEvent(*(evt->GetEvent()));
  
  int nTauMatched =0, ntaujet=0,nelec=0,nmuon=0;
  bool event_passed = true;
  
  TLorentzVector taunet,tauelec,taumuon;
  HepMC::GenEvent::particle_iterator p;
  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    
    
    if(abs((*p)->pdg_id()) == 15&&(*p)->status()==2) { 
      bool lept_decay = false;
      
      //Searches for Taus
      //HepLorentzVector partMom = ( *p )->momentum() ;
      //if(partMom.et() < mEtTau) event_passed = false;
      //if(fabs(partMom.eta()) > mEtaMax) event_passed = false;
      
      TLorentzVector tau((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e());
      HepMC::GenVertex::particle_iterator z = (*p)->end_vertex()->particles_begin(HepMC::descendants);
      for(; z != (*p)->end_vertex()->particles_end(HepMC::descendants); z++)
	{
	  if(abs((*z)->pdg_id()) == 11 || abs((*z)->pdg_id()) == 13)lept_decay=true;
	  if(abs((*z)->pdg_id()) == 11)
	    {
	      nelec++;
	      tauelec.SetPxPyPzE((*z)->momentum().px(),(*z)->momentum().py(),(*z)->momentum().pz(),(*z)->momentum().e());
	      if((*z)->momentum().perp()<mPtElec)event_passed = false;
	      if(fabs((*z)->momentum().eta())>mEtaElecMax)event_passed = false;
	    }
	  if(abs((*z)->pdg_id()) == 13)
	    {
	      nmuon++;
	      taumuon.SetPxPyPzE((*z)->momentum().px(),(*z)->momentum().py(),(*z)->momentum().pz(),(*z)->momentum().e());
	      if((*z)->momentum().perp()<mPtMuon)event_passed = false;
	      if(fabs((*z)->momentum().eta())>mEtaMuonMax)event_passed = false;

	    }
	  if(abs((*z)->pdg_id()) == 16)taunet.SetPxPyPzE((*z)->momentum().px(),(*z)->momentum().py(),(*z)->momentum().pz(),(*z)->momentum().e());
	  
	}
      if(lept_decay==false)
	{
	  ntaujet++;
	  TLorentzVector jetMom=tau-taunet;
	  if(jetMom.Et() < mEtTau) event_passed = false;
	  if(fabs(jetMom.Eta()) > mEtaMax) event_passed = false;
	  
	}
      
      /*
	if(event_passed)
	{
	L1JetParticleCollection::const_iterator l1Tau = myL1Tau.begin();
	for(;l1Tau != myL1Tau.end();l1Tau++)
	{
	math::XYZVector candDir((*p)->momentum().x(),(*p)->momentum().y(),(*p)->momentum().z());
	double deltaR = ROOT::Math::VectorUtil::DeltaR((*l1Tau).momentum(), candDir);
	if(deltaR < 0.15) nTauMatched++;
	
	
	}
	
	}
      */
    }
    
  }
  
  
  string decay_type;
  if(nelec==1&&ntaujet==1&&nmuon==0)decay_type="etau";
  if(nelec==0&&ntaujet==1&&nmuon==1)decay_type="mutau";
  if(nelec==0&&ntaujet==2&&nmuon==0)decay_type="tautau";
  if(nelec==1&&ntaujet==0&&nmuon==1)decay_type="emu";
  if(nelec==2&&ntaujet==0&&nmuon==0)decay_type="ee";
  if(nelec==0&&ntaujet==0&&nmuon==2)decay_type="mumu";
  
  bool decay=false;
  for(vstring::const_iterator e = mincludeList.begin();e != mincludeList.end(); ++ e ) 
    {     
      if((*e)==decay_type)decay=true;
    }
  
  //if(mn_taujet!=ntaujet||mn_elec!=nelec||mn_muon!=nmuon)event_passed=false;
  
  //if(event_passed&&mn_taujet==ntaujet&&mn_elec==nelec&&mn_muon==nmuon) cout <<"TauJet::"<<ntaujet<<"::TauElec::"<<nelec<<"::TauMuon::"<<nmuon<<endl;
  delete generated_event; 
  return (event_passed&&decay);
  

}
