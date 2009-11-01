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

  L1_Collection_  = parameters_.getUntrackedParameter<InputTag>("L1_Collection", edm::InputTag("hltL1extraParticles"));
  L2_Collection_  = parameters_.getUntrackedParameter<InputTag>("L2_Collection", edm::InputTag("hltL2MuonCandidates"));
  L3_Collection_  = parameters_.getUntrackedParameter<InputTag>("L3_Collection", edm::InputTag("hltL3MuonCandidates"));

  dbe_ = Service<DQMStore>().operator->();

}


TopHLTDiMuonDQM::~TopHLTDiMuonDQM() {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginJob(const EventSetup& context) {

  dbe_ = Service<DQMStore>().operator->();

  if( dbe_ ) {

    dbe_->setCurrentFolder("monitorName_");
    if( monitorName_ != "" )  monitorName_ = monitorName_+"/" ;
    if( verbose_ )  cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;

    dbe_->setCurrentFolder(monitorName_+level_);

    Trigs = dbe_->book1D("HLTDimuon_Trigs", "Fired triggers", 10, 0., 10.);
    Trigs->setAxisTitle("Fired triggers", 1);

    NMuons = dbe_->book1D("HLTDimuon_NMuons", "Number of muons", 10, 0., 10.);
    NMuons->setAxisTitle("Number of muons", 1);

    PtMuons = dbe_->book1D("HLTDimuon_Pt","P_T of muons", 100, 0., 200.);
    PtMuons->setAxisTitle("P^{muon}_{T}  (GeV)", 1);

    EtaMuons = dbe_->book1D("HLTDimuon_Eta","Pseudorapidity of muons", 100, -5., 5.);
    EtaMuons->setAxisTitle("#eta_{muon}", 1);

    PhiMuons = dbe_->book1D("HLTDimuon_Phi","Azimutal angle of muons", 70, -3.5, 3.5);
    PhiMuons->setAxisTitle("#phi_{muon}  (rad)", 1);

    DiMuonMass = dbe_->book1D("HLTDimuon_DiMuonMass","Invariant Dimuon Mass", 100, 0., 200.);
    DiMuonMass->setAxisTitle("Invariant #mu #mu mass  (GeV)", 1);

    // define logarithmic bins for a histogram with 100 bins going from 10^0 to 10^3

    const int nbins = 100;

    double logmin = 0.;
    double logmax = 3.;

    float bins[nbins+1];

    for (int i = 0; i <= nbins; i++) {

      double log = logmin + (logmax-logmin)*i/nbins;
      bins[i] = std::pow(10.0, log);

    }

    DiMuonMass_LOG = dbe_->book1D("HLTDimuon_DiMuonMass_LOG","Invariant Dimuon Mass", nbins, &bins[0]);
    DiMuonMass_LOG->setAxisTitle("Invariant #mu #mu mass  (GeV)", 1);

    DeltaEtaMuons = dbe_->book1D("HLTDimuon_DeltaEta","#Delta #eta of muon pair", 100, -5., 5.);
    DeltaEtaMuons->setAxisTitle("#Delta #eta_{#mu #mu}", 1);

    DeltaPhiMuons = dbe_->book1D("HLTDimuon_DeltaPhi","#Delta #phi of muon pair", 100, -5., 5.);
    DeltaPhiMuons->setAxisTitle("#Delta #phi_{#mu #mu}  (rad)", 1);
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

    cout << endl << "-----------------------------" << endl;
    cout << "--- NO TRIGGER RESULTS !! ---" << endl;
    cout << "-----------------------------" << endl << endl;

  }

  const int n_TrigPaths = hltPaths.size();

  bool FiredTriggers[100] = {false};

  if( !trigResults.failedToGet() ) {

    int n_Triggers = trigResults->size();

    TriggerNames trigName;
    trigName.init(*trigResults);

    for( int i_Trig = 0; i_Trig < n_Triggers; ++i_Trig ) {

      if (trigResults.product()->accept(i_Trig)) {

	for( int i = 0; i < n_TrigPaths; i++ ) {

	  if ( trigName.triggerName(i_Trig)== hltPaths[i] ) {

	    FiredTriggers[i] = true;
	    Trigs->Fill(i);

	    //	    cout << "--------------------" << endl;
	    //	    cout << "Trigger: " << hltPaths[i] << " FIRED!!!  " << endl;
	    //	    cout << "-----------------------------" << endl << endl;

	  }

	}

      }

    }

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

      cout << endl << "------------------------------" << endl;
      cout << "--- NO L1 TRIGGER MUONS !! ---" << endl;
      cout << "------------------------------" << endl << endl;

    }

    if( !mucands.failedToGet() ) {

      NMuons->Fill(mucands->size());

      cout << "--------------------" << endl;
      cout << " Nmuons: " << mucands->size() << endl;
      cout << "--------------------" << endl << endl;

      L1MuonParticleCollection::const_iterator cand;

      for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

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

      }

    }

  }

  if( level_ == "L3" ) {

    Handle<RecoChargedCandidateCollection> mucands;
    iEvent.getByLabel(L3_Collection_, mucands);

    if( mucands.failedToGet() ) {

      cout << endl << "-----------------------------" << endl;
      cout << "--- NO HLTRIGGER MUONS !! ---" << endl;
      cout << "-----------------------------" << endl << endl;

    }

    if( !mucands.failedToGet() ) {

      NMuons->Fill(mucands->size());

      cout << "--------------------" << endl;
      cout << " Nmuons: " << mucands->size() << endl;
      cout << "--------------------" << endl << endl;

      RecoChargedCandidateCollection::const_iterator cand;

      for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

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

      }

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
 
  //  if( outputFile_.size() != 0 && dbe_ )  dbe_->save(outputFile_);
  return;

}
