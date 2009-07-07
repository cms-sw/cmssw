// Original Author: S. Gennai

// Modified by Author: K. Petridis S.Greder

#include "HLTriggerOffline/Tau/interface/TauJetMCFilter.h"

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

  _fillHistos = iConfig.getParameter<bool>("fillHistos");
  _doPrintOut = iConfig.getParameter<bool>("doPrintOut");

  ////////////////////////////////////////////////////////  
  // Book histograms
  if (_fillHistos) {
    edm::Service<TFileService> fs;
    TFileDirectory dir = fs->mkdir("histos");
  
    h_TauEt  = dir.make<TH1F>("TauEt", "E_{T}",60,0.,60.); 
    h_TauEt->Sumw2();
    h_TauEta  = dir.make<TH1F>("TauEta", "#eta",40,-5.,5.); 
    h_TauEta->Sumw2();
    h_TauPhi  = dir.make<TH1F>("TauPhi", "#phi",40,-3.2,3.2); 
    h_TauPhi->Sumw2();
    
    h_MuonPt  = dir.make<TH1F>("MuonPt", "P_{T}",60,0.,60.); 
    h_MuonPt->Sumw2();
    h_MuonEta  = dir.make<TH1F>("MuonEta", "#eta",40,-5.,5.); 
    h_MuonEta->Sumw2();
    h_MuonPhi  = dir.make<TH1F>("MuonPhi", "#phi",40,-3.2,3.2); 
    h_MuonPhi->Sumw2();
    
    h_ElecEt  = dir.make<TH1F>("ElecEt", "E_{T}",60,0.,60.); 
    h_ElecEt->Sumw2();
    h_ElecEta  = dir.make<TH1F>("ElecEta", "#eta",40,-5.,5.); 
    h_ElecEta->Sumw2();
    h_ElecPhi  = dir.make<TH1F>("ElecPhi", "#phi",40,-3.2,3.2); 
    h_ElecPhi->Sumw2();
  }
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
  
  _nEvents++;

  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(genParticles, evt);
  
  HepMC::GenEvent * generated_event = new HepMC::GenEvent(*(evt->GetEvent()));
  
  //int nTauMatched =0;
  int ntaujet=0,nelec=0,nmuon=0;
  bool event_passed = true;

  bool passedElecEtaCut = true;
  bool passedElecEtCut = true;
  bool passedMuonEtaCut = true;
  bool passedMuonPtCut = true;
  bool passedTauEtaCut = true;
  bool passedTauEtCut = true;
  
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
	      if (_fillHistos) {
		h_ElecEt->Fill((*z)->momentum().perp()); 
		h_ElecEta->Fill((*z)->momentum().eta()); 
		h_ElecPhi->Fill((*z)->momentum().phi()); 
	      }
	      
	      if(fabs((*z)->momentum().eta())>mEtaElecMax) {
		event_passed = false;
		passedElecEtaCut = false;
	      }	
	      if((*z)->momentum().perp()<mPtElec) {
		event_passed = false;
		passedElecEtCut = false;
	      }
	    }
	  if(abs((*z)->pdg_id()) == 13)
	    {
	      nmuon++;
	      taumuon.SetPxPyPzE((*z)->momentum().px(),(*z)->momentum().py(),(*z)->momentum().pz(),(*z)->momentum().e());

	      if (_fillHistos) {
		h_MuonPt->Fill((*z)->momentum().perp()); 
		h_MuonEta->Fill((*z)->momentum().eta()); 
		h_MuonPhi->Fill((*z)->momentum().phi()); 
	      }
	      
	      if(fabs((*z)->momentum().eta())>mEtaMuonMax) {
		event_passed = false;
		passedMuonEtaCut = false;
	      }
	      if((*z)->momentum().perp()<mPtMuon) {
		event_passed = false;
		passedMuonPtCut = false;
	      }
	    }
	  if(abs((*z)->pdg_id()) == 16)taunet.SetPxPyPzE((*z)->momentum().px(),(*z)->momentum().py(),(*z)->momentum().pz(),(*z)->momentum().e());
	  
	}
      if(lept_decay==false)
	{
	  ntaujet++;
	  TLorentzVector jetMom=tau-taunet;

	  if (_fillHistos) {
	    h_TauEt->Fill(jetMom.Et()); 
	    h_TauEta->Fill(jetMom.Eta()); 
	    h_TauPhi->Fill(jetMom.Phi()); 
	  }
	  
	  if(fabs(jetMom.Eta()) > mEtaMax) {
	    event_passed = false;
	    passedTauEtaCut = false;
	  }
	  if(jetMom.Et() < mEtTau) {
	    event_passed = false;
	    passedTauEtCut = false;
	  }
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

  /*  
  if (passedMuonPtCut && passedMuonEtaCut && passedElecEtaCut) {
    _nPassedElecEtaCut++;
  }
  if (passedMuonPtCut && passedMuonEtaCut && passedElecEtaCut && passedElecEtCut) {
    _nPassedElecEtCut++;
  }
  */

  if (ntaujet>=1) {
    _nPassednTauCut++;
  }
  if (ntaujet>=1 && passedTauEtaCut) {
    _nPassedTauEtaCut++;
  }
  if (ntaujet>=1 && passedTauEtaCut && passedTauEtCut) {
    _nPassedTauEtCut++;
  }
  if (ntaujet>=1 && passedTauEtaCut && passedTauEtCut 
      && nmuon>=1) {
    _nPassednMuonCut++;
  }
  if (ntaujet>=1 && passedTauEtaCut && passedTauEtCut
      && nmuon>=1 && passedMuonEtaCut) {
    _nPassedMuonEtaCut++;
  }
  if (ntaujet>=1 && passedTauEtaCut && passedTauEtCut
      && nmuon>=1 && passedMuonPtCut && passedMuonEtaCut) {
    _nPassedMuonPtCut++;
  }

  if(event_passed&&decay)
    _nPassedAllCuts++;
  
  //if(mn_taujet!=ntaujet||mn_elec!=nelec||mn_muon!=nmuon)event_passed=false;
  
  //if(event_passed&&mn_taujet==ntaujet&&mn_elec==nelec&&mn_muon==nmuon) cout <<"TauJet::"<<ntaujet<<"::TauElec::"<<nelec<<"::TauMuon::"<<nmuon<<endl;
  delete generated_event; 
  return (event_passed&&decay);
  

}



// ------------ method called once each job just before starting event loop  ------------
void 
TauJetMCFilter::beginJob(const edm::EventSetup&)
{
  _nEvents = 0;
  _nPassedElecEtaCut = 0;
  _nPassedElecEtCut = 0;
  _nPassedMuonEtaCut = 0;
  _nPassedMuonPtCut = 0;
  _nPassedTauEtaCut = 0;
  _nPassedTauEtCut = 0;
  _nPassedAllCuts = 0;

  _nPassednMuonCut = 0;
  _nPassednTauCut = 0;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TauJetMCFilter::endJob() {
  using namespace std;

  if (_doPrintOut) {
    string MarkerString = "###MuTauMCFilter### ";
    cout<<MarkerString<<"*** Start: MuTauMCFilter Efficiency Report ***"<<endl;
    cout<<MarkerString<<"nEvents processed             : "<<_nEvents<<endl;
    cout<<MarkerString<<"------------------------------------------------------------"<<endl;
    cout<<MarkerString<<"nEvents passing ntaujet cut   : "<<_nPassednTauCut<<endl;
    cout<<MarkerString<<"nEvents passing tau eta cut   : "<<_nPassedTauEtaCut<<endl;
    cout<<MarkerString<<"nEvents passing tau et cut    : "<<_nPassedTauEtCut<<endl;
    cout<<MarkerString<<"nEvents passing nmuon cut     : "<<_nPassednMuonCut<<endl;
    cout<<MarkerString<<"nEvents passing muon eta cut  : "<<_nPassedMuonEtaCut<<endl;
    cout<<MarkerString<<"nEvents passing muon pt cut   : "<<_nPassedMuonPtCut<<endl;
    cout<<MarkerString<<"------------------------------------------------------------"<<endl;
    cout<<MarkerString<<"nEvents passing all cuts      : "<<_nPassedAllCuts<<endl;
    cout<<MarkerString<<"*** End: MuTauMCFilter Efficiency Report ***"<<endl;
    cout<<endl;
  }
}
 
