/*
 * Package:  TopHLTDiMuonDQM
 *   Class:  TopHLTDiMuonDQM
 *
 * Original Author:  Muriel VANDER DONCKT *:0
 *         Created:  Wed Dec 12 09:55:42 CET 2007
 *   Original Code:  HLTMuonRecoDQMSource.cc,v 1.2 2008/10/16 16:41:29 hdyoo Exp $
 *
 */

#include "DQMOffline/Trigger/interface/TopHLTDiMuonDQM.h"
#include "FWCore/Common/interface/TriggerNames.h"

using namespace std;
using namespace edm;
using namespace trigger;

//
// constructors and destructor
//

TopHLTDiMuonDQM::TopHLTDiMuonDQM( const edm::ParameterSet& ps ) {

  level_       = ps.getUntrackedParameter<string>("Level", "TEV");
  monitorName_ = ps.getUntrackedParameter<string>("monitorName", "HLT/Top/HLTDiMuon/");

  triggerResults_ = ps.getParameter<edm::InputTag>("TriggerResults");
  triggerEvent_   = ps.getParameter<edm::InputTag>("TriggerEvent");
  triggerFilter_  = ps.getParameter<edm::InputTag>("TriggerFilter");

  hltPaths_L1_   = ps.getParameter<vector<string> >("hltPaths_L1");
  hltPaths_L3_   = ps.getParameter<vector<string> >("hltPaths_L3");
  hltPaths_sig_  = ps.getParameter<vector<string> >("hltPaths_sig");
  hltPaths_trig_ = ps.getParameter<vector<string> >("hltPaths_trig");

  L1_Collection_ = ps.getUntrackedParameter<edm::InputTag>("L1_Collection", InputTag("hltL1extraParticles"));
  L3_Collection_ = ps.getUntrackedParameter<edm::InputTag>("L3_Collection", InputTag("hltL3MuonCandidates"));
  L3_Isolation_  = ps.getUntrackedParameter<edm::InputTag>("L3_Isolation",  InputTag("hltL3MuonIsolations"));

  vertex_       = ps.getParameter<edm::InputTag>("vertexCollection");
  vertex_X_cut_ = ps.getParameter<double>("vertex_X_cut");
  vertex_Y_cut_ = ps.getParameter<double>("vertex_Y_cut");
  vertex_Z_cut_ = ps.getParameter<double>("vertex_Z_cut");

  muons_        = ps.getParameter<edm::InputTag>("muonCollection");
  muon_pT_cut_  = ps.getParameter<double>("muon_pT_cut");
  muon_eta_cut_ = ps.getParameter<double>("muon_eta_cut");
  muon_iso_cut_ = ps.getParameter<double>("muon_iso_cut");

  MassWindow_up_   = ps.getParameter<double>("MassWindow_up");
  MassWindow_down_ = ps.getParameter<double>("MassWindow_down");

  for(int i=0; i<100; ++i) {
    N_sig[i]  = 0;
    N_trig[i] = 0;
    Eff[i]    = 0.;
  }

}


TopHLTDiMuonDQM::~TopHLTDiMuonDQM() {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginJob() {

  dbe_ = Service<DQMStore>().operator->();

  if( dbe_ ) {

    dbe_->setCurrentFolder(monitorName_+level_);

    Trigs = dbe_->book1D("01_Trigs", "Fired triggers", 15, 0., 15.);

    TriggerEfficiencies = dbe_->book1D("02_TriggerEfficiencies", "HL Trigger Efficiencies", 10, 0., 10.);
    TriggerEfficiencies->setTitle("HL Trigger Efficiencies #epsilon_{signal} = #frac{[signal] && [control]}{[control]}");

    NMuons        = dbe_->book1D("05_Nmuons",        "Number of muons",             20,   0.,  10.);
    NMuons_iso    = dbe_->book1D("06_Nmuons_Iso",    "Number of isolated muons",    20,   0.,  10.);
    NMuons_charge = dbe_->book1D("13_Nmuons_Charge", "N_{muons} * Q(#mu)",          19, -10.,  10.);
    NTracks       = dbe_->book1D("Ntracks",          "Number of tracks",            50,   0.,  50.);
    VxVy_muons    = dbe_->book2D("VxVy_muons",       "Vertex x-y-positon (global)", 40,  -1.,   1., 40 , -1., 1.);
    Vz_muons      = dbe_->book1D("Vz_muons",         "Vertex z-positon (global)",   40, -20.,  20.);
    PtMuons       = dbe_->book1D("15_Pt_muon",       "P^{#mu}_{T}",                 20,   0., 200.);
    EtaMuons      = dbe_->book1D("18_Eta_muon",      "#eta_{muon}",                 20,  -5.,   5.);
    PhiMuons      = dbe_->book1D("14_Phi_muon",      "#phi_{muon}",                 20,  -4.,   4.);
    DeltaEtaMuons = dbe_->book1D("11_DeltaEta",      "#Delta #eta of muon pair",    20,  -5.,   5.);
    DeltaPhiMuons = dbe_->book1D("12_DeltaPhi",      "#Delta #phi of muon pair",    20,  -4.,   4.);
    CombRelIso03  = dbe_->book1D("07_MuIso_CombRelIso03", "Muon CombRelIso dR=03",  20,   0.,   1.);

    PtMuons_sig   = dbe_->book1D("16_Pt_sig",   "P^{#mu}_{T} (signal triggered)",   20,  0., 200.);
    PtMuons_trig  = dbe_->book1D("17_Pt_trig",  "P^{#mu}_{T} (control triggered)",  20,  0., 200.);
    EtaMuons_sig  = dbe_->book1D("19_Eta_sig",  "#eta_{muon} (signal triggered)",   20, -5.,   5.);
    EtaMuons_trig = dbe_->book1D("20_Eta_trig", "#eta_{muon} (control triggered)",  20, -5.,   5.);

    MuonEfficiency_pT  = dbe_->book1D("07_MuonEfficiency_pT",  "Muon Efficiency P_{T}", 20,  0., 200.);
    MuonEfficiency_eta = dbe_->book1D("08_MuonEfficiency_eta", "Muon Efficiency  #eta", 20, -5.,   5.);

    const int nbins = 200;

    double logmin = 0.;
    double logmax = 3.;  // 10^(3.)=1000

    float bins[nbins+1];

    for (int i = 0; i <= nbins; i++) {

      double log = logmin + (logmax-logmin)*i/nbins;
      bins[i] = std::pow(10.0, log);

    }

    DiMuonMassRC       = dbe_->book1D("03_DiMuonMass_RC",       "Invariant Dimuon Mass (Right Charge)", 50, 0., 200.);
    DiMuonMassRC_LOGX  = dbe_->book1D("04_DiMuonMass_RC_LOGX",  "Invariant Dimuon Mass (Right Charge)", nbins, &bins[0]);
    DiMuonMassRC_LOG10 = dbe_->book1D("21_DiMuonMass_RC_LOG10", "Invariant Dimuon Mass (Right Charge)", 50, 0., 2.5);

    DiMuonMassWC       = dbe_->book1D("09_DiMuonMass_WC",       "Invariant Dimuon Mass (Wrong Charge)", 50, 0., 200.);
    DiMuonMassWC_LOGX  = dbe_->book1D("10_DiMuonMass_WC_LOGX",  "Invariant Dimuon Mass (Wrong Charge)", nbins, &bins[0]);
    DiMuonMassWC_LOG10 = dbe_->book1D("22_DiMuonMass_WC_LOG10", "Invariant Dimuon Mass (Wrong Charge)", 50, 0., 2.5);

  }

} 


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginRun(const edm::Run& r, const edm::EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {

}


// ----------------------------------------------------------
void TopHLTDiMuonDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  // ------------------------
  //  Global Event Variables
  // ------------------------

  vector<string> hltPaths;

  if( level_ == "L1"   )  hltPaths = hltPaths_L1_;
  if( level_ == "TEV"  )  hltPaths = hltPaths_L1_;
  if( level_ == "L3"   )  hltPaths = hltPaths_L3_;
  if( level_ == "RECO" )  hltPaths = hltPaths_L3_;

  const int N_TriggerPaths = hltPaths.size();
  const int N_SignalPaths  = hltPaths_sig_.size();
  const int N_ControlPaths = hltPaths_trig_.size();

  bool Fired_Signal_Trigger[100]  = {false};
  bool Fired_Control_Trigger[100] = {false};

  double DilepMass = 0.;

  // -------------------------
  //  Analyze Trigger Results
  // -------------------------

  edm::Handle<TriggerResults> trigResults;
  iEvent.getByLabel(triggerResults_, trigResults);

  if( trigResults.failedToGet() ) {

    //    cout << endl << "-----------------------------" << endl;
    //    cout << "--- NO TRIGGER RESULTS !! ---" << endl;
    //    cout << "-----------------------------" << endl << endl;

  }

  if( !trigResults.failedToGet() ) {

    int N_Triggers = trigResults->size();

    const edm::TriggerNames & trigName = iEvent.triggerNames(*trigResults);

    for( int i_Trig = 0; i_Trig < N_Triggers; ++i_Trig ) {

      if(trigResults.product()->accept(i_Trig)) {

	// Check for all trigger paths

	for( int i = 0; i < N_TriggerPaths; i++ ) {

	  if( trigName.triggerName(i_Trig) == hltPaths[i] ) {

	    Trigs->Fill(i);
	    Trigs->setBinLabel( i+1, hltPaths[i], 1);

	  }

	}

	// Check for signal & control trigger paths

	for( int j = 0; j < N_SignalPaths; ++j ) {

	  if( trigName.triggerName(i_Trig) == hltPaths_sig_[j]  )  Fired_Signal_Trigger[j]  = true;

	}

	for( int k = 0; k < N_ControlPaths; ++k ) {

	  if( trigName.triggerName(i_Trig) == hltPaths_trig_[k] )  Fired_Control_Trigger[k] = true;

	}

      }

    }

  }

  // -----------------------
  //  Analyze Trigger Muons
  // -----------------------

  // -------------------
  //  From TriggerEvent
  // -------------------

  if( level_ == "TEV" ) {

    edm::Handle<TriggerEvent> triggerEvent;
    iEvent.getByLabel(triggerEvent_, triggerEvent);

    if( triggerEvent.failedToGet() ) {

      //      cout << endl << "---------------------------" << endl;
      //      cout << "--- NO TRIGGER EVENT !! ---" << endl;
      //      cout << "---------------------------" << endl << endl;

    }

    if( !triggerEvent.failedToGet() ) {

      size_t filterIndex = triggerEvent->filterIndex( triggerFilter_ );
      TriggerObjectCollection triggerObjects = triggerEvent->getObjects();

      TriggerObjectCollection::const_iterator trig;

      if( filterIndex < triggerEvent->sizeFilters() ) {

	const Keys & keys = triggerEvent->filterKeys( filterIndex );

	NMuons->Fill(keys.size());

	int N_mu = 0;

	for( size_t j = 0; j < keys.size(); j++ ) {

	  TriggerObject foundObject = triggerObjects[keys[j]];

	  reco::Particle cand = foundObject.particle();

	  float N_muons = keys.size();
	  float Q_muon  = cand.charge();

	  NMuons_charge->Fill(N_muons*Q_muon);

	  if(     cand.pt()   < muon_pT_cut_  )  continue;
	  if( abs(cand.eta()) > muon_eta_cut_ )  continue;
	  //	  if( cand.isIsolated() )  ++N_iso_mu;

	  ++N_mu;

	}

	if( N_mu > 1 ) {

	  reco::Particle mu1 = triggerObjects[keys[0]].particle();
	  reco::Particle mu2 = triggerObjects[keys[1]].particle();

	  DilepMass = sqrt( (mu1.energy() + mu2.energy())*(mu1.energy() + mu2.energy())
			    - (mu1.px() + mu2.px())*(mu1.px() + mu2.px())
			    - (mu1.py() + mu2.py())*(mu1.py() + mu2.py())
			    - (mu1.pz() + mu2.pz())*(mu1.pz() + mu2.pz()) );

	  DiMuonMassRC_LOG10->Fill( log10(DilepMass) );
	  DiMuonMassRC->Fill(             DilepMass  );
	  DiMuonMassRC_LOGX->Fill(        DilepMass  );

	  if( DilepMass > MassWindow_down_ && DilepMass < MassWindow_up_ ) {

	    for( size_t j = 0; j < keys.size(); j++ ) {

	      TriggerObject  foundObject = triggerObjects[keys[j]];
	      reco::Particle        cand = foundObject.particle();

	      PtMuons->Fill(  cand.pt()  );
	      EtaMuons->Fill( cand.eta() );
	      PhiMuons->Fill( cand.phi() );

	    }

	    DeltaEtaMuons->Fill( mu1.eta()-mu2.eta() );
	    DeltaPhiMuons->Fill( mu1.phi()-mu2.phi() );

	  }

	}

      }

    }

  }

  // -------------------------
  //  From L1 Muon Collection
  // -------------------------

  if( level_ == "L1" ) {

    edm::Handle<l1extra::L1MuonParticleCollection> mucands;
    iEvent.getByLabel(L1_Collection_, mucands);

    if( mucands.failedToGet() ) {

      //      cout << endl << "------------------------------" << endl;
      //      cout << "--- NO L1 TRIGGER MUONS !! ---" << endl;
      //      cout << "------------------------------" << endl << endl;

    }

    if( !mucands.failedToGet() ) {

      NMuons->Fill(mucands->size());

      l1extra::L1MuonParticleCollection::const_iterator cand;

      int N_iso_mu = 0;

      for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

	float N_muons = mucands->size();
	float Q_muon  = cand->charge();

	NMuons_charge->Fill(N_muons*Q_muon);

	if(     cand->pt()   < muon_pT_cut_  )  continue;
	if( abs(cand->eta()) > muon_eta_cut_ )  continue;
	if( cand->isIsolated() )  ++N_iso_mu;

      }

      NMuons_iso->Fill(N_iso_mu);

      //      if( N_iso_mu > 1 && Fired_Control_Trigger[0] ) {
      if( N_iso_mu > 1 ) {

	l1extra::L1MuonParticleCollection::const_reference mu1 = mucands->at(0);
	l1extra::L1MuonParticleCollection::const_reference mu2 = mucands->at(1);

	DilepMass = sqrt( (mu1.energy() + mu2.energy())*(mu1.energy() + mu2.energy())
			  - (mu1.px() + mu2.px())*(mu1.px() + mu2.px())
			  - (mu1.py() + mu2.py())*(mu1.py() + mu2.py())
			  - (mu1.pz() + mu2.pz())*(mu1.pz() + mu2.pz()) );

	// Opposite muon charges -> Right Charge (RC)

	if( mu1.charge()*mu2.charge() < 0. ) {

	  DiMuonMassRC_LOG10->Fill( log10(DilepMass) );
	  DiMuonMassRC->Fill(             DilepMass  );
	  DiMuonMassRC_LOGX->Fill(        DilepMass  );

	  if( DilepMass > MassWindow_down_ && DilepMass < MassWindow_up_ ) {

	    for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

	      PtMuons->Fill(  cand->pt()  );
	      EtaMuons->Fill( cand->eta() );
	      PhiMuons->Fill( cand->phi() );

	    }

	    DeltaEtaMuons->Fill( mu1.eta()-mu2.eta() );
	    DeltaPhiMuons->Fill( mu1.phi()-mu2.phi() );

	    // Determinating trigger efficiencies

	    for( int k = 0; k < N_SignalPaths; ++k ) {

	      if( Fired_Signal_Trigger[k] && Fired_Control_Trigger[k] )  ++N_sig[k];

	      if( Fired_Control_Trigger[k] )  ++N_trig[k];

	      if( N_trig[k] != 0 )  Eff[k] = N_sig[k]/static_cast<float>(N_trig[k]);

	      TriggerEfficiencies->setBinContent( k+1, Eff[k] );
	      TriggerEfficiencies->setBinLabel( k+1, "#frac{["+hltPaths_sig_[k]+"]}{vs. ["+hltPaths_trig_[k]+"]}", 1);

	    }

	    if( Fired_Signal_Trigger[0] && Fired_Control_Trigger[0] ) {

	      PtMuons_sig->Fill(mu1.pt());
	      EtaMuons_sig->Fill(mu1.eta());

	    }

	    if( Fired_Control_Trigger[0] ) {

	      PtMuons_trig->Fill(mu1.pt());
	      EtaMuons_trig->Fill(mu1.eta());

	    }

	  }

	}

	// Same muon charges -> Wrong Charge (WC)

	if( mu1.charge()*mu2.charge() > 0. ) {

	  DiMuonMassWC_LOG10->Fill( log10(DilepMass) );
	  DiMuonMassWC->Fill(             DilepMass  );
	  DiMuonMassWC_LOGX->Fill(        DilepMass  );

	}

      }

    }

  }

  // -------------------------
  //  From L3 Muon Collection
  // -------------------------

  if( level_ == "L3" ) {

    edm::Handle<reco::RecoChargedCandidateCollection> mucands;
    iEvent.getByLabel(L3_Collection_, mucands);

    if( mucands.failedToGet() ) {

      //      cout << endl << "------------------------------" << endl;
      //      cout << "--- NO HL TRIGGER MUONS !! ---" << endl;
      //      cout << "------------------------------" << endl << endl;

    }

    edm::Handle<ValueMap<bool> > isoMap;
    iEvent.getByLabel(L3_Isolation_, isoMap);

    if( isoMap.failedToGet() ) {

      //      cout << endl << "--------------------------------" << endl;
      //      cout << "--- NO MUON ISOLATION MAP !! ---" << endl;
      //      cout << "--------------------------------" << endl << endl;

    }

    if( !mucands.failedToGet() ) {

      NMuons->Fill(mucands->size());

      reco::RecoChargedCandidateCollection::const_iterator cand;

      int N_iso_mu = 0;

      for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

	float N_muons = mucands->size();
	float Q_muon  = cand->charge();

	NMuons_charge->Fill(N_muons*Q_muon);

	double track_X = 100.;
	double track_Y = 100.;
	double track_Z = 100.;

	reco::TrackRef track = cand->track();

	track_X = track->vx();
	track_Y = track->vy();
	track_Z = track->vz();

	// Vertex and kinematic cuts

	if(          track_X > vertex_X_cut_ )  continue;
	if(          track_Y > vertex_Y_cut_ )  continue;
	if(          track_Z > vertex_Z_cut_ )  continue;
	if(     cand->pt()   < muon_pT_cut_  )  continue;
	if( abs(cand->eta()) > muon_eta_cut_ )  continue;

	if( isoMap.isValid() ) {

	  // Isolation flag (this is a bool value: true => isolated)
	  ValueMap<bool>::value_type muonIsIsolated = (*isoMap)[track];

	  if( muonIsIsolated )  ++N_iso_mu;

	}

      }

      NMuons_iso->Fill(N_iso_mu);

      //      if( N_iso_mu > 1 && Fired_Control_Trigger[0] ) {
      if( N_iso_mu > 1 ) {

	reco::RecoChargedCandidateCollection::const_reference mu1 = mucands->at(0);
	reco::RecoChargedCandidateCollection::const_reference mu2 = mucands->at(1);

	DilepMass = sqrt( (mu1.energy() + mu2.energy())*(mu1.energy() + mu2.energy())
			  - (mu1.px() + mu2.px())*(mu1.px() + mu2.px())
			  - (mu1.py() + mu2.py())*(mu1.py() + mu2.py())
			  - (mu1.pz() + mu2.pz())*(mu1.pz() + mu2.pz()) );

	// Opposite muon charges -> Right Charge (RC)

	if( mu1.charge()*mu2.charge() < 0. ) {

	  DiMuonMassRC_LOG10->Fill( log10(DilepMass) );
	  DiMuonMassRC->Fill(             DilepMass  );
	  DiMuonMassRC_LOGX->Fill(        DilepMass  );

	  if( DilepMass > MassWindow_down_ && DilepMass < MassWindow_up_ ) {

	    for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

	      PtMuons->Fill(  cand->pt()  );
	      EtaMuons->Fill( cand->eta() );
	      PhiMuons->Fill( cand->phi() );

	    }

	    DeltaEtaMuons->Fill( mu1.eta()-mu2.eta() );
	    DeltaPhiMuons->Fill( mu1.phi()-mu2.phi() );

	    // Determinating trigger efficiencies

	    for( int k = 0; k < N_SignalPaths; ++k ) {

	      if( Fired_Signal_Trigger[k] && Fired_Control_Trigger[k] )  ++N_sig[k];

	      if( Fired_Control_Trigger[k] )  ++N_trig[k];

	      if( N_trig[k] != 0 )  Eff[k] = N_sig[k]/static_cast<float>(N_trig[k]);

	      TriggerEfficiencies->setBinContent( k+1, Eff[k] );
	      TriggerEfficiencies->setBinLabel( k+1, "#frac{["+hltPaths_sig_[k]+"]}{vs. ["+hltPaths_trig_[k]+"]}", 1);

	    }

	    if( Fired_Signal_Trigger[0] && Fired_Control_Trigger[0] ) {

	      PtMuons_sig->Fill(mu1.pt());
	      EtaMuons_sig->Fill(mu1.eta());

	    }

	    if( Fired_Control_Trigger[0] ) {

	      PtMuons_trig->Fill(mu1.pt());
	      EtaMuons_trig->Fill(mu1.eta());

	    }

	  }

	}

	// Same muon charges -> Wrong Charge (WC)

	if( mu1.charge()*mu2.charge() > 0. ) {

	  DiMuonMassWC_LOG10->Fill( log10(DilepMass) );
	  DiMuonMassWC->Fill(             DilepMass  );
	  DiMuonMassWC_LOGX->Fill(        DilepMass  );

	}

      }

    }

  }

  // ---------------------------
  //  From RECO Muon Collection
  // ---------------------------

  if( level_ == "RECO" ) {

    int N_iso_mu  = 0;

    double vertex_X  = 100.;
    double vertex_Y  = 100.;
    double vertex_Z  = 100.;

    // Analyze Primary Vertex

    edm::Handle<reco::VertexCollection> vertexs;
    iEvent.getByLabel(vertex_, vertexs);

    if( vertexs.failedToGet() ) {

      //      cout << endl << "----------------------------" << endl;
      //      cout << "--- NO PRIMARY VERTEX !! ---" << endl;
      //      cout << "----------------------------" << endl << endl;

    }

    if( !vertexs.failedToGet() ) {

      reco::Vertex primaryVertex = vertexs->front();

      int numberTracks = primaryVertex.tracksSize();
      //      double ndof      = primaryVertex.ndof();
      bool fake        = primaryVertex.isFake();

      NTracks->Fill(numberTracks);

      if( !fake && numberTracks > 3 ) {

	vertex_X = primaryVertex.x();
	vertex_Y = primaryVertex.y();
	vertex_Z = primaryVertex.z();

      }

    }

    // Analyze Muon Isolation

    edm::Handle<reco::MuonCollection> muons;
    iEvent.getByLabel(muons_, muons);

    reco::MuonCollection::const_iterator muon;

    if( muons.failedToGet() ) {

      //      cout << endl << "------------------------" << endl;
      //      cout << "--- NO RECO MUONS !! ---" << endl;
      //      cout << "------------------------" << endl << endl;

    }

    if( !muons.failedToGet() ) {

      NMuons->Fill( muons->size() );

      for(muon = muons->begin(); muon!= muons->end(); ++muon) {

	float N_muons = muons->size();
	float Q_muon  = muon->charge();

	NMuons_charge->Fill(N_muons*Q_muon);

	double track_X = 100.;
	double track_Y = 100.;
	double track_Z = 100.;

	if( muon->isGlobalMuon() ) {

	  reco::TrackRef track = muon->globalTrack();

	  track_X = track->vx();
	  track_Y = track->vy();
	  track_Z = track->vz();

	  VxVy_muons->Fill(track_X, track_Y);
	  Vz_muons->Fill(track_Z);

	}

	// Vertex and kinematic cuts

	if(          track_X > vertex_X_cut_ )  continue;
	if(          track_Y > vertex_Y_cut_ )  continue;
	if(          track_Z > vertex_Z_cut_ )  continue;
	if(     muon->pt()   < muon_pT_cut_  )  continue;
	if( abs(muon->eta()) > muon_eta_cut_ )  continue;

	reco::MuonIsolation muIso03 = muon->isolationR03();

	double muonCombRelIso = 1.;

	if ( muon->pt() != 0. )
	  muonCombRelIso = ( muIso03.emEt + muIso03.hadEt + muIso03.hoEt + muIso03.sumPt ) / muon->pt();

	CombRelIso03->Fill( muonCombRelIso );

	if( muonCombRelIso < muon_iso_cut_ )  ++N_iso_mu;

      }

      NMuons_iso->Fill(N_iso_mu);

      //      if( N_iso_mu > 1 && Fired_Control_Trigger[0] ) {
      if( N_iso_mu > 1 ) {

	// Vertex cut

	if( vertex_X < vertex_X_cut_ && vertex_Y < vertex_Y_cut_ && vertex_Z < vertex_Z_cut_ ) {

	  reco::MuonCollection::const_reference mu1 = muons->at(0);
	  reco::MuonCollection::const_reference mu2 = muons->at(1);

	  DilepMass = sqrt( (mu1.energy()+mu2.energy())*(mu1.energy()+mu2.energy())
			    - (mu1.px()+mu2.px())*(mu1.px()+mu2.px())
			    - (mu1.py()+mu2.py())*(mu1.py()+mu2.py())
			    - (mu1.pz()+mu2.pz())*(mu1.pz()+mu2.pz())
			    );

	  // Opposite muon charges -> Right Charge (RC)

	  if( mu1.charge()*mu2.charge() < 0. ) {

	    DiMuonMassRC_LOG10->Fill( log10(DilepMass) );
	    DiMuonMassRC->Fill(      DilepMass );
	    DiMuonMassRC_LOGX->Fill( DilepMass );

	    if( DilepMass > MassWindow_down_ && DilepMass < MassWindow_up_ ) {

	      for(muon = muons->begin(); muon!= muons->end(); ++muon) {

		PtMuons->Fill(  muon->pt()  );
		EtaMuons->Fill( muon->eta() );
		PhiMuons->Fill( muon->phi() );

	      }

	      DeltaEtaMuons->Fill(mu1.eta()-mu2.eta());
	      DeltaPhiMuons->Fill(mu1.phi()-mu2.phi());

	      // Determinating trigger efficiencies

	      for( int k = 0; k < N_SignalPaths; ++k ) {

		if( Fired_Signal_Trigger[k] && Fired_Control_Trigger[k] )  ++N_sig[k];

		if( Fired_Control_Trigger[k] )  ++N_trig[k];

		if( N_trig[k] != 0 )  Eff[k] = N_sig[k]/static_cast<float>(N_trig[k]);

		TriggerEfficiencies->setBinContent( k+1, Eff[k] );
		TriggerEfficiencies->setBinLabel( k+1, "#frac{["+hltPaths_sig_[k]+"]}{vs. ["+hltPaths_trig_[k]+"]}", 1);

	      }

	      if( Fired_Signal_Trigger[0] && Fired_Control_Trigger[0] ) {

		PtMuons_sig->Fill(mu1.pt());
		EtaMuons_sig->Fill(mu1.eta());

	      }

	      if( Fired_Control_Trigger[0] ) {

		PtMuons_trig->Fill(mu1.pt());
		EtaMuons_trig->Fill(mu1.eta());

	      }

	    }

	  }

	  // Same muon charges -> Wrong Charge (WC)

	  if( mu1.charge()*mu2.charge() > 0. ) {

	    DiMuonMassWC_LOG10->Fill( log10(DilepMass) );
	    DiMuonMassWC->Fill(      DilepMass );
	    DiMuonMassWC_LOGX->Fill( DilepMass );

	  }

	}

      }

    }

  }

  const int N_bins_pT  = PtMuons_trig->getNbinsX();
  const int N_bins_eta = EtaMuons_trig->getNbinsX();

  for( int i = 0; i < N_bins_pT; ++i ) {

    if( PtMuons_trig->getBinContent(i) != 0 ) {

      Eff[0] = PtMuons_sig->getBinContent(i)/PtMuons_trig->getBinContent(i);
      MuonEfficiency_pT->setBinContent( i, Eff[0] );

    }

  }

  for( int j = 0; j < N_bins_eta; ++j ) {

    if( EtaMuons_trig->getBinContent(j) != 0 ) {

      Eff[0] = EtaMuons_sig->getBinContent(j)/EtaMuons_trig->getBinContent(j);
      MuonEfficiency_eta->setBinContent( j, Eff[0] );

    }

  }

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::endRun(const edm::Run& r, const edm::EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::endJob() {

}
