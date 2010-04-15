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

//
// constructors and destructor
//

TopHLTDiMuonDQM::TopHLTDiMuonDQM( const ParameterSet& parameters_ ) : counterEvt_( 0 ) {

  verbose_        = parameters_.getUntrackedParameter<bool>("verbose", false);
  monitorName_    = parameters_.getUntrackedParameter<string>("monitorName", "Top/HLTDiMuons");
  prescaleEvt_    = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);

  level_          = parameters_.getUntrackedParameter<string>("Level", "L3");
  triggerResults_ = parameters_.getParameter<InputTag>("TriggerResults");
  hltPaths_L1_    = parameters_.getParameter<vector<string> >("hltPaths_L1");
  hltPaths_L3_    = parameters_.getParameter<vector<string> >("hltPaths_L3");
  hltPaths_sig_   = parameters_.getParameter<vector<string> >("hltPaths_sig");
  hltPaths_trig_  = parameters_.getParameter<vector<string> >("hltPaths_trig");

  L1_Collection_  = parameters_.getUntrackedParameter<InputTag>("L1_Collection", InputTag("hltL1extraParticles"));
  L3_Collection_  = parameters_.getUntrackedParameter<InputTag>("L3_Collection", InputTag("hltL3MuonCandidates"));
  L3_Isolation_   = parameters_.getUntrackedParameter<InputTag>("L3_Isolation",  InputTag("hltL3MuonIsolations"));

  muon_pT_cut_    = parameters_.getParameter<double>("muon_pT_cut");
  muon_eta_cut_   = parameters_.getParameter<double>("muon_eta_cut");

  MassWindow_up_   = parameters_.getParameter<double>("MassWindow_up");
  MassWindow_down_ = parameters_.getParameter<double>("MassWindow_down");

  //  dbe_ = Service<DQMStore>().operator->();

  for(int i=0; i<100; ++i) {
    N_sig[i]  = 0;
    N_trig[i] = 0;
    Eff[i]    = 0.;
  }

}


TopHLTDiMuonDQM::~TopHLTDiMuonDQM() {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginJob(void) {

  dbe_ = Service<DQMStore>().operator->();

  if( dbe_ ) {

    dbe_->setCurrentFolder("monitorName_");
    if( monitorName_ != "" )  monitorName_ = monitorName_+"/" ;
    if( verbose_ )  cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;

    dbe_->setCurrentFolder(monitorName_+level_);

    Trigs = dbe_->book1D("01_HLTDimuon_Trigs", "Fired triggers", 10, 0., 10.);
    Trigs->setAxisTitle("", 1);

    TriggerEfficiencies = dbe_->book1D("02_HLTDimuon_TriggerEfficiencies", "HL Trigger Efficiencies", 5, 0., 5.);
    //    TriggerEfficiencies->setAxisTitle("#epsilon_{signal} = #frac{[signal] && [control]}{[control]}", 1);
    TriggerEfficiencies->setTitle("HL Trigger Efficiencies #epsilon_{signal} = #frac{[signal] && [control]}{[control]}");

    NMuons = dbe_->book1D("05_HLTDimuon_NMuons", "Number of muons", 20, 0., 10.);
    NMuons->setAxisTitle("Number of muons", 1);

    NMuons_iso = dbe_->book1D("06_HLTDimuon_NMuons_Iso", "Number of isolated muons", 20, 0., 10.);
    NMuons_iso->setAxisTitle("", 1);

    MuonEfficiency_pT = dbe_->book1D("07_HLTDimuon_MuonEfficiency_pT","Muon Efficiency P_{T}", 20, 0., 200.);
    MuonEfficiency_pT->setAxisTitle("P^{#mu}_{T}  (GeV)", 1);

    MuonEfficiency_eta = dbe_->book1D("08_HLTDimuon_MuonEfficiency_eta","Muon Efficiency  #eta", 20, -5., 5.);
    MuonEfficiency_eta->setAxisTitle("#eta_{#mu}", 1);

    DeltaEtaMuons = dbe_->book1D("11_HLTDimuon_DeltaEta","#Delta #eta of muon pair", 20, -5., 5.);
    DeltaEtaMuons->setAxisTitle("#Delta #eta_{#mu #mu}", 1);

    DeltaPhiMuons = dbe_->book1D("12_HLTDimuon_DeltaPhi","#Delta #phi of muon pair", 20, -4., 4.);
    DeltaPhiMuons->setAxisTitle("#Delta #phi_{#mu #mu}  (rad)", 1);

    NMuons_charge = dbe_->book1D("13_HLTDimuon_NMuons_Charge", "Number of muons * Moun charge", 19, -10., 10.);
    NMuons_charge->setAxisTitle("N_{muons} * Q(#mu)", 1);

    PhiMuons = dbe_->book1D("14_HLTDimuon_Phi","Azimutal angle of muons", 20, -4., 4.);
    PhiMuons->setAxisTitle("#phi_{muon}  (rad)", 1);

    PtMuons = dbe_->book1D("15_HLTDimuon_Pt","P_T of muons", 20, 0., 200.);
    PtMuons->setAxisTitle("P^{#mu}_{T}  (GeV)", 1);

    PtMuons_sig = dbe_->book1D("16_HLTDimuon_Pt_sig","P_T of signal triggered muons", 20, 0., 200.);
    PtMuons_sig->setAxisTitle("P^{#mu}_{T} (signal triggered)  (GeV)", 1);

    PtMuons_trig = dbe_->book1D("17_HLTDimuon_Pt_trig","P_T of control triggered muons", 20, 0., 200.);
    PtMuons_trig->setAxisTitle("P^{#mu}_{T} (control triggered)  (GeV)", 1);

    EtaMuons = dbe_->book1D("18_HLTDimuon_Eta","Pseudorapidity of muons", 20, -5., 5.);
    EtaMuons->setAxisTitle("#eta_{muon}", 1);

    EtaMuons_sig = dbe_->book1D("19_HLTDimuon_Eta_sig","Pseudorapidity of signal triggered muons", 20, -5., 5.);
    EtaMuons_sig->setAxisTitle("#eta_{muon} (signal triggered)", 1);

    EtaMuons_trig = dbe_->book1D("20_HLTDimuon_Eta_trig","Pseudorapidity of control triggered muons", 20, -5., 5.);
    EtaMuons_trig->setAxisTitle("#eta_{muon} (control triggered)", 1);

    const int nbins = 50;

    double logmin = 0.;
    double logmax = 3.;  // 10^(3.)=1000

    float bins[nbins+1];

    for (int i = 0; i <= nbins; i++) {

      double log = logmin + (logmax-logmin)*i/nbins;
      bins[i] = std::pow(10.0, log);

    }

    DiMuonMassRC = dbe_->book1D("03_HLTDimuon_DiMuonMass_RC","Invariant Dimuon Mass (Right Charge)", 50, 0., 200.);
    DiMuonMassRC->setAxisTitle("Invariant #mu #mu mass  (GeV)", 1);

    DiMuonMassRC_LOGX = dbe_->book1D("04_HLTDimuon_DiMuonMass_RC_LOGX","Invariant Dimuon Mass (Right Charge)", nbins, &bins[0]);
    DiMuonMassRC_LOGX->setAxisTitle("LOG_10[ Invariant #mu #mu mass (GeV) ]", 1);

    DiMuonMassRC_LOG10 = dbe_->book1D("21_HLTDimuon_DiMuonMass_RC_LOG10","Invariant Dimuon Mass (Right Charge)", 50, 0., 2.5);
    DiMuonMassRC_LOG10->setAxisTitle("LOG_10[ Invariant #mu #mu mass (GeV) ]", 1);

    DiMuonMassWC = dbe_->book1D("09_HLTDimuon_DiMuonMass_WC","Invariant Dimuon Mass (Wrong Charge)", 50, 0., 200.);
    DiMuonMassWC->setAxisTitle("Invariant #mu #mu mass  (GeV)", 1);

    DiMuonMassWC_LOGX = dbe_->book1D("10_HLTDimuon_DiMuonMass_WC_LOGX","Invariant Dimuon Mass (Wrong Charge)", nbins, &bins[0]);
    DiMuonMassWC_LOGX->setAxisTitle("LOG_10[ Invariant #mu #mu mass (GeV) ]", 1);

    DiMuonMassWC_LOG10 = dbe_->book1D("22_HLTDimuon_DiMuonMass_WC_LOG10","Invariant Dimuon Mass (Wrong Charge)", 50, 0., 2.5);
    DiMuonMassWC_LOG10->setAxisTitle("LOG_10[ Invariant #mu #mu mass (GeV) ]", 1);

  }

} 


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginRun(const Run& r, const EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

}


// ----------------------------------------------------------
void TopHLTDiMuonDQM::analyze(const Event& iEvent, const EventSetup& iSetup ) {

  if( !dbe_ ) return;

  counterEvt_++;

  // ------------------------
  //  Global Event Variables
  // ------------------------

  vector<string> hltPaths;

  if( level_ == "L1" )  hltPaths = hltPaths_L1_;

  if( level_ == "L3" )  hltPaths = hltPaths_L3_;

  const int N_TriggerPaths = hltPaths.size();
  const int N_SignalPaths  = hltPaths_sig_.size();
  const int N_ControlPaths = hltPaths_trig_.size();

  bool Fired_Signal_Trigger[ 100] = {false};
  bool Fired_Control_Trigger[100] = {false};

  double DilepMass = 0.;

  // -------------------------
  //  Analyze Trigger Results
  // -------------------------

  Handle<TriggerResults> trigResults;
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

	    //	    cout << "Trigger: " << hltPaths[i] << " FIRED!!! " << endl;

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

  if( level_ == "L1" ) {

    Handle<l1extra::L1MuonParticleCollection> mucands;
    iEvent.getByLabel(L1_Collection_, mucands);

    if( mucands.failedToGet() ) {

      //      cout << endl << "------------------------------" << endl;
      //      cout << "--- NO L1 TRIGGER MUONS !! ---" << endl;
      //      cout << "------------------------------" << endl << endl;

    }

    if( !mucands.failedToGet() ) {

      NMuons->Fill(mucands->size());

      //      cout << "---------------" << endl;
      //      cout << " Nmuons: " << mucands->size() << endl;
      //      cout << "---------------" << endl << endl;

      l1extra::L1MuonParticleCollection::const_iterator cand;

      int N_iso_mu = 0;

      for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

	float N_muons = mucands->size();
	float Q_muon  = cand->charge();

	NMuons_charge->Fill(N_muons*Q_muon);

	if( cand->isIsolated() )  ++N_iso_mu;

      }

      //      cout << "Nmuons_iso: " << N_iso_mu << endl;

      NMuons_iso->Fill(N_iso_mu);


      if( N_iso_mu > 1 && Fired_Control_Trigger[0] ) {

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

	      if(     cand->pt()   < muon_pT_cut_     )  continue;
	      if( abs(cand->eta()) > muon_eta_cut_    )  continue;

	      PtMuons->Fill(  cand->pt()  );
	      EtaMuons->Fill( cand->eta() );
	      PhiMuons->Fill( cand->phi() );

	    }

	    DeltaEtaMuons->Fill( mu1.eta()-mu2.eta() );
	    DeltaPhiMuons->Fill( mu1.phi()-mu2.phi() );

	    // Determinating trigger efficiencies

	    //	    cout << "-----------------------------"   << endl;

	    for( int k = 0; k < N_SignalPaths; ++k ) {

	      if( Fired_Signal_Trigger[k] && Fired_Control_Trigger[k] )  ++N_sig[k];

	      if( Fired_Control_Trigger[k] )  ++N_trig[k];

	      if( N_trig[k] != 0 )  Eff[k] = N_sig[k]/static_cast<float>(N_trig[k]);

	      //	      cout << "Signal Trigger  : " << hltPaths_sig_[k]  << "\t: " << N_sig[k]  << endl;
	      //	      cout << "Control Trigger : " << hltPaths_trig_[k] << "\t: " << N_trig[k] << endl;
	      //	      cout << "Trigger Eff.cy  : " << Eff[k]  << endl;
	      //	      cout << "-----------------------------" << endl;

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

  if( level_ == "L3" ) {

    Handle<reco::RecoChargedCandidateCollection> mucands;
    iEvent.getByLabel(L3_Collection_, mucands);

    if( mucands.failedToGet() ) {

      //      cout << endl << "------------------------------" << endl;
      //      cout << "--- NO HL TRIGGER MUONS !! ---" << endl;
      //      cout << "------------------------------" << endl << endl;

    }

    Handle<ValueMap<bool> > isoMap;
    iEvent.getByLabel(L3_Isolation_, isoMap);

    if( isoMap.failedToGet() ) {

      //      cout << endl << "--------------------------------" << endl;
      //      cout << "--- NO MUON ISOLATION MAP !! ---" << endl;
      //      cout << "--------------------------------" << endl << endl;

    }

    if( !mucands.failedToGet() ) {

      NMuons->Fill(mucands->size());

      //      cout << "---------------" << endl;
      //      cout << " Nmuons: " << mucands->size() << endl;
      //      cout << "---------------" << endl << endl;

      reco::RecoChargedCandidateCollection::const_iterator cand;

      int N_iso_mu = 0;

      for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

	float N_muons = mucands->size();
	float Q_muon  = cand->charge();

	NMuons_charge->Fill(N_muons*Q_muon);

	reco::TrackRef tk = cand->track();

	if( isoMap.isValid() ) {

	  // Isolation flag (this is a bool value: true => isolated)
	  ValueMap<bool>::value_type muonIsIsolated = (*isoMap)[tk];

	  if( muonIsIsolated )  ++N_iso_mu;

	}

      }

      //      cout << "Nmuons_iso: " << N_iso_mu << endl;

      NMuons_iso->Fill(N_iso_mu);


      if( N_iso_mu > 1 && Fired_Control_Trigger[0] ) {

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

	      if(     cand->pt()   < muon_pT_cut_     )  continue;
	      if( abs(cand->eta()) > muon_eta_cut_    )  continue;

	      PtMuons->Fill(  cand->pt()  );
	      EtaMuons->Fill( cand->eta() );
	      PhiMuons->Fill( cand->phi() );

	    }

	    DeltaEtaMuons->Fill( mu1.eta()-mu2.eta() );
	    DeltaPhiMuons->Fill( mu1.phi()-mu2.phi() );

	    // Determinating trigger efficiencies

	    //	    cout << "-----------------------------"   << endl;

	    for( int k = 0; k < N_SignalPaths; ++k ) {

	      if( Fired_Signal_Trigger[k] && Fired_Control_Trigger[k] )  ++N_sig[k];

	      if( Fired_Control_Trigger[k] )  ++N_trig[k];

	      if( N_trig[k] != 0 )  Eff[k] = N_sig[k]/static_cast<float>(N_trig[k]);

	      //	      cout << "Signal Trigger  : " << hltPaths_sig_[k]  << "\t: " << N_sig[k]  << endl;
	      //	      cout << "Control Trigger : " << hltPaths_trig_[k] << "\t: " << N_trig[k] << endl;
	      //	      cout << "Trigger Eff.cy  : " << Eff[k]  << endl;
	      //	      cout << "-----------------------------" << endl;

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
void TopHLTDiMuonDQM::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::endRun(const Run& r, const EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::endJob() {

  LogInfo("HLTMonMuon") << "analyzed " << counterEvt_ << " events";

  return;

}

// Declare this as an analyzer for the Framework
DEFINE_FWK_MODULE(TopHLTDiMuonDQM);
