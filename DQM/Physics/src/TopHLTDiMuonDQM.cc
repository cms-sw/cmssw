/*
 * Package:  TopHLTDiMuonDQM
 *   Class:  TopHLTDiMuonDQM
 *
 * Original Author:  Muriel VANDER DONCKT *:0
 *         Created:  Wed Dec 12 09:55:42 CET 2007
 *   Original Code:  HLTMuonRecoDQMSource.cc,v 1.2 2008/10/16 16:41:29 hdyoo Exp $
 *
 */

#include "DQM/Physics/src/TopHLTDiMuonDQM.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;

//
// constructors and destructor
//

TopHLTDiMuonDQM::TopHLTDiMuonDQM( const ParameterSet& parameters_ ) : counterEvt_( 0 )
{

  verbose_        = parameters_.getUntrackedParameter<bool>("verbose", false);
  monitorName_    = parameters_.getUntrackedParameter<string>("monitorName", "Top/HLTDiMuons");
  prescaleEvt_    = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);

  level_          = parameters_.getUntrackedParameter<string>("Level", "L3");
  triggerResults_ = parameters_.getParameter<InputTag>("TriggerResults");
  hltPaths_L1_    = parameters_.getParameter<vector<string> >("hltPaths_L1");
  hltPaths_L3_    = parameters_.getParameter<vector<string> >("hltPaths_L3");
  hltPath_sig_    = parameters_.getParameter<vector<string> >("hltPath_sig");
  hltPath_trig_   = parameters_.getParameter<vector<string> >("hltPath_trig");

  L1_Collection_  = parameters_.getUntrackedParameter<InputTag>("L1_Collection", edm::InputTag("hltL1extraParticles"));
  L2_Collection_  = parameters_.getUntrackedParameter<InputTag>("L2_Collection", edm::InputTag("hltL2MuonCandidates"));
  L3_Collection_  = parameters_.getUntrackedParameter<InputTag>("L3_Collection", edm::InputTag("hltL3MuonCandidates"));

  muon_pT_cut_    = parameters_.getParameter<double>("muon_pT_cut");
  muon_eta_cut_   = parameters_.getParameter<double>("muon_eta_cut");

  dbe_ = Service<DQMStore>().operator->();

  N_sig  = 0;
  N_trig = 0;
  eff    = 0.;

}


TopHLTDiMuonDQM::~TopHLTDiMuonDQM() {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginJob(const EventSetup& context) {

  //  dbe_ = Service<DQMStore>().operator->();

  if( dbe_ ) {

    dbe_->setCurrentFolder("monitorName_");
    if( monitorName_ != "" )  monitorName_ = monitorName_+"/" ;
    if( verbose_ )  cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;

    dbe_->setCurrentFolder(monitorName_+level_);

    Trigs = dbe_->book1D("HLTDimuon_Trigs", "Fired triggers", 10, 0., 10.);
    Trigs->setAxisTitle("Fired triggers", 1);

    NMuons = dbe_->book1D("HLTDimuon_NMuons", "Number of muons", 10, 0., 10.);
    NMuons->setAxisTitle("Number of muons", 1);

    PtMuons = dbe_->book1D("HLTDimuon_Pt","P_T of muons", 50, 0., 200.);
    PtMuons->setAxisTitle("P^{#mu}_{T}  (GeV)", 1);

    PtMuons_sig = dbe_->book1D("HLTDimuon_Pt_sig","P_T of signal triggered muons", 50, 0., 200.);
    PtMuons_sig->setAxisTitle("P^{#mu}_{T} (signal triggered)  (GeV)", 1);

    PtMuons_trig = dbe_->book1D("HLTDimuon_Pt_trig","P_T of control triggered muons", 50, 0., 200.);
    PtMuons_trig->setAxisTitle("P^{#mu}_{T} (control triggered)  (GeV)", 1);

    EtaMuons = dbe_->book1D("HLTDimuon_Eta","Pseudorapidity of muons", 50, -5., 5.);
    EtaMuons->setAxisTitle("#eta_{muon}", 1);

    EtaMuons_sig = dbe_->book1D("HLTDimuon_Eta_sig","Pseudorapidity of signal triggered muons", 50, -5., 5.);
    EtaMuons_sig->setAxisTitle("#eta_{muon} (signal triggered)", 1);

    EtaMuons_trig = dbe_->book1D("HLTDimuon_Eta_trig","Pseudorapidity of control triggered muons", 50, -5., 5.);
    EtaMuons_trig->setAxisTitle("#eta_{muon} (control triggered)", 1);

    PhiMuons = dbe_->book1D("HLTDimuon_Phi","Azimutal angle of muons", 70, -3.5, 3.5);
    PhiMuons->setAxisTitle("#phi_{muon}  (rad)", 1);

    DiMuonMass = dbe_->book1D("HLTDimuon_DiMuonMass","Invariant Dimuon Mass", 50, 0., 500.);
    DiMuonMass->setAxisTitle("Invariant #mu #mu mass  (GeV)", 1);

    // define logarithmic bins for a histogram with 100 bins going from 10^0 to 10^3

    const int nbins = 50;

    double logmin = 0.;
    double logmax = 2.7;  // 10^(2.7)=~500

    float bins[nbins+1];

    for (int i = 0; i <= nbins; i++) {

      double log = logmin + (logmax-logmin)*i/nbins;
      bins[i] = std::pow(10.0, log);

    }

    DiMuonMass_LOG = dbe_->book1D("HLTDimuon_DiMuonMass_LOG","Invariant Dimuon Mass", nbins, &bins[0]);
    DiMuonMass_LOG->setAxisTitle("Invariant #mu #mu mass  (GeV)", 1);

    DeltaEtaMuons = dbe_->book1D("HLTDimuon_DeltaEta","#Delta #eta of muon pair", 50, -5., 5.);
    DeltaEtaMuons->setAxisTitle("#Delta #eta_{#mu #mu}", 1);

    DeltaPhiMuons = dbe_->book1D("HLTDimuon_DeltaPhi","#Delta #phi of muon pair", 50, -5., 5.);
    DeltaPhiMuons->setAxisTitle("#Delta #phi_{#mu #mu}  (rad)", 1);

    MuonEfficiency_pT = dbe_->book1D("HLTDimuon_MuonEfficiency_pT","Muon Efficiency P_{T}", 50, 0., 200.);
    MuonEfficiency_pT->setAxisTitle("P^{#mu}_{T}  (GeV)", 1);

    MuonEfficiency_eta = dbe_->book1D("HLTDimuon_MuonEfficiency_eta","Muon Efficiency  #eta", 50, -5., 5.);
    MuonEfficiency_eta->setAxisTitle("#eta_{#mu}", 1);

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

  // -------------------------
  //  Analyze Trigger Results
  // -------------------------

  vector<string> hltPaths;

  if( level_ == "L1" )  hltPaths = hltPaths_L1_;

  if( level_ == "L3" )  hltPaths = hltPaths_L3_;

  Handle<TriggerResults> trigResults;
  iEvent.getByLabel(triggerResults_, trigResults);

  if( trigResults.failedToGet() ) {

    //    cout << endl << "-----------------------------" << endl;
    //    cout << "--- NO TRIGGER RESULTS !! ---" << endl;
    //    cout << "-----------------------------" << endl << endl;

  }

  const int n_TrigPaths = hltPaths.size();

  bool FiredTriggers[100]    = {false};
  bool Fired_Signal_Trigger  =  false;
  bool Fired_Control_Trigger =  false;

  if( !trigResults.failedToGet() ) {

    int n_Triggers = trigResults->size();

    TriggerNames trigName;
    trigName.init(*trigResults);

    for( int i_Trig = 0; i_Trig < n_Triggers; ++i_Trig ) {

      if(trigResults.product()->accept(i_Trig)) {

	if( trigName.triggerName(i_Trig) == hltPath_sig_[0]  )  Fired_Signal_Trigger  = true;

	if( trigName.triggerName(i_Trig) == hltPath_trig_[0] )  Fired_Control_Trigger = true;

	for( int i = 0; i < n_TrigPaths; i++ ) {

	  if( trigName.triggerName(i_Trig) == hltPaths[i] ) {

	    FiredTriggers[i] = true;
	    Trigs->Fill(i);

	    //	    cout << "-----------------------------" << endl;
	    //	    cout << "Trigger: " << hltPaths[i] << " FIRED!!!  " << endl;
	    //	    cout << "-----------------------------" << endl << endl;

	  }

	}

      }

    }

    if( Fired_Signal_Trigger && Fired_Control_Trigger )  ++N_sig;

    if( Fired_Control_Trigger )  ++N_trig;

    //    cout << "-----------------------------" << endl;
    //    cout << "Signal Trigger  : " << N_sig  << endl;
    //    cout << "Control Trigger : " << N_trig << endl;
    //    cout << "-----------------------------" << endl << endl;

  }

  // -----------------------
  //  Analyze Trigger Muons
  // -----------------------

  //  Handle<L1MuonParticleCollection> mucands_L1;
  //  Handle<RecoChargedCandidateCollection> mucands_L3;

  if( level_ == "L1" ) {

    Handle<L1MuonParticleCollection> mucands;
    iEvent.getByLabel(L1_Collection_, mucands);

    if( mucands.failedToGet() ) {

      //      cout << endl << "------------------------------" << endl;
      //      cout << "--- NO L1 TRIGGER MUONS !! ---" << endl;
      //      cout << "------------------------------" << endl << endl;

    }

    if( !mucands.failedToGet() ) {

      NMuons->Fill(mucands->size());

      //      cout << "--------------------" << endl;
      //      cout << " Nmuons: " << mucands->size() << endl;
      //      cout << "--------------------" << endl << endl;

      L1MuonParticleCollection::const_iterator cand;

      for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

	if(     cand->pt()   < muon_pT_cut_  )  continue;
	if( abs(cand->eta()) > muon_eta_cut_ )  continue;
	if( !Fired_Signal_Trigger )             continue;

	PtMuons->Fill(  cand->pt()  );
	EtaMuons->Fill( cand->eta() );
	PhiMuons->Fill( cand->phi() );

      }

      if( mucands->size() > 1 ) {

	L1MuonParticleCollection::const_reference mu1 = mucands->at(0);
	L1MuonParticleCollection::const_reference mu2 = mucands->at(1);

	//      cout << "-----------------------------" << endl;
	//      cout << "Muon_1 Pt: " << mu1.pt() << endl;
	//      cout << "Muon_2 Pt: " << mu2.pt() << endl;
	//      cout << "-----------------------------" << endl << endl;

	DeltaEtaMuons->Fill( mu1.eta()-mu2.eta() );
	DeltaPhiMuons->Fill( mu1.phi()-mu2.phi() );

	double dilepMass = sqrt( (mu1.energy() + mu2.energy())*(mu1.energy() + mu2.energy())
				 - (mu1.px() + mu2.px())*(mu1.px() + mu2.px())
				 - (mu1.py() + mu2.py())*(mu1.py() + mu2.py())
				 - (mu1.pz() + mu2.pz())*(mu1.pz() + mu2.pz()) );

	DiMuonMass_LOG->Fill( dilepMass );
	DiMuonMass->Fill( dilepMass );

	if( Fired_Signal_Trigger && Fired_Control_Trigger ) {

	  PtMuons_sig->Fill(mu1.pt());
	  EtaMuons_sig->Fill(mu1.eta());

	}

	if( Fired_Control_Trigger ) {

	  PtMuons_trig->Fill(mu1.pt());
	  EtaMuons_trig->Fill(mu1.eta());

	}

      }

    }

  }

  if( level_ == "L3" ) {

    Handle<RecoChargedCandidateCollection> mucands;
    iEvent.getByLabel(L3_Collection_, mucands);

    if( mucands.failedToGet() ) {

      //      cout << endl << "------------------------------" << endl;
      //      cout << "--- NO HL TRIGGER MUONS !! ---" << endl;
      //      cout << "------------------------------" << endl << endl;

    }

    if( !mucands.failedToGet() ) {

      NMuons->Fill(mucands->size());

      //      cout << "--------------------" << endl;
      //      cout << " Nmuons: " << mucands->size() << endl;
      //      cout << "--------------------" << endl << endl;

      RecoChargedCandidateCollection::const_iterator cand;

      for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

	if(     cand->pt()   < muon_pT_cut_  )  continue;
	if( abs(cand->eta()) > muon_eta_cut_ )  continue;
	if( !Fired_Signal_Trigger )             continue;

	PtMuons->Fill(  cand->pt()  );
	EtaMuons->Fill( cand->eta() );
	PhiMuons->Fill( cand->phi() );

      }

      if( mucands->size() > 1 ) {

	RecoChargedCandidateCollection::const_reference mu1 = mucands->at(0);
	RecoChargedCandidateCollection::const_reference mu2 = mucands->at(1);

	//      cout << "-----------------------------" << endl;
	//      cout << "Muon_1 Pt: " << mu1.pt() << endl;
	//      cout << "Muon_2 Pt: " << mu2.pt() << endl;
	//      cout << "-----------------------------" << endl << endl;

	DeltaEtaMuons->Fill( mu1.eta()-mu2.eta() );
	DeltaPhiMuons->Fill( mu1.phi()-mu2.phi() );

	double dilepMass = sqrt( (mu1.energy() + mu2.energy())*(mu1.energy() + mu2.energy())
				 - (mu1.px() + mu2.px())*(mu1.px() + mu2.px())
				 - (mu1.py() + mu2.py())*(mu1.py() + mu2.py())
				 - (mu1.pz() + mu2.pz())*(mu1.pz() + mu2.pz()) );

	DiMuonMass_LOG->Fill( dilepMass );
	DiMuonMass->Fill( dilepMass );

	if( Fired_Signal_Trigger && Fired_Control_Trigger ) {

	  PtMuons_sig->Fill(mu1.pt());
	  EtaMuons_sig->Fill(mu1.eta());

	}

	if( Fired_Control_Trigger ) {

	  PtMuons_trig->Fill(mu1.pt());
	  EtaMuons_trig->Fill(mu1.eta());

	}

      }

    }

  }

  const int N_bins_pT  = PtMuons_trig->getNbinsX();
  const int N_bins_eta = EtaMuons_trig->getNbinsX();

  for( int i = 0; i < N_bins_pT; ++i ) {

    if( PtMuons_trig->getBinContent(i) != 0 ) {

      eff = PtMuons_sig->getBinContent(i)/PtMuons_trig->getBinContent(i);
      MuonEfficiency_pT->setBinContent( i, eff );

    }

  }

  for( int j = 0; j < N_bins_eta; ++j ) {

    if( EtaMuons_trig->getBinContent(j) != 0 ) {

      eff = EtaMuons_sig->getBinContent(j)/EtaMuons_trig->getBinContent(j);
      MuonEfficiency_eta->setBinContent( j, eff );

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
