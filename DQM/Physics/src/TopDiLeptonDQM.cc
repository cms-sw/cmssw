/*
 *  $Date: 2009/07/29 13:05:04 $
 *  $Revision: 1.2 $
 *  \author M. Marienfeld - DESY Hamburg
 */

#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DQM/Physics/src/TopDiLeptonDQM.h"

using namespace std;
using namespace edm;

TopDiLeptonDQM::TopDiLeptonDQM( const edm::ParameterSet& ps ) {

  parameters_ = ps;
  initialize();

  moduleName_     = ps.getUntrackedParameter<string>("moduleName");
  triggerResults_ = ps.getParameter<InputTag>("TriggerResults");
  hltPaths_L3_    = ps.getParameter<vector<string> >("hltPaths_L3");
  hltPaths_L3_mu_ = ps.getParameter<vector<string> >("hltPaths_L3_mu");
  hltPaths_L3_el_ = ps.getParameter<vector<string> >("hltPaths_L3_el");

  muons_          = ps.getParameter<edm::InputTag>("muonCollection");
  muon_pT_cut_    = ps.getParameter<double>("muon_pT_cut");
  muon_eta_cut_   = ps.getParameter<double>("muon_eta_cut");

  elecs_          = ps.getParameter<edm::InputTag>("elecCollection");
  elec_pT_cut_    = ps.getParameter<double>("elec_pT_cut");
  elec_eta_cut_   = ps.getParameter<double>("elec_eta_cut");

}


TopDiLeptonDQM::~TopDiLeptonDQM() {

}


void TopDiLeptonDQM::initialize() {

}


void TopDiLeptonDQM::beginJob(const edm::EventSetup& evt) {

  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder(moduleName_);

  Trigs_ = dbe_->book1D("Trigs", "Fired muon/electron triggers", 10,  0.,  10.);

  Muon_Trigs_ = dbe_->book1D("Muon_Trigs", "Fired muon triggers",  10,  0.,  10.);
  Nmuons_     = dbe_->book1D("Nmuons",     "Nmuons",               10,  0.,  10.);
  pT_muons_   = dbe_->book1D("pT_muons",   "pT_muons",            100,  0., 200.);
  eta_muons_  = dbe_->book1D("eta_muons",  "eta_muons",           100, -5.,   5.);
  phi_muons_  = dbe_->book1D("phi_muons",  "phi_muons",            80, -4.,   4.);

  Elec_Trigs_ = dbe_->book1D("Elec_Trigs", "Fired electron triggers",  10,  0.,  10.);
  Nelecs_     = dbe_->book1D("Nelecs",     "Nelecs",                   10,  0.,  10.);
  pT_elecs_   = dbe_->book1D("pT_elecs",   "pT_elecs",                100,  0., 200.);
  eta_elecs_  = dbe_->book1D("eta_elecs",  "eta_elecs",               100, -5.,   5.);
  phi_elecs_  = dbe_->book1D("phi_elecs",  "phi_elecs",                80, -4.,   4.);

  MuIso_emEt03_    = dbe_->book1D("MuIso_emEt03",    "Muon emEt03",    25, 0., 25.);
  MuIso_hadEt03_   = dbe_->book1D("MuIso_hadEt03",   "Muon hadEt03",   20, 0., 25.);
  MuIso_hoEt03_    = dbe_->book1D("MuIso_hoEt03",    "Muon hoEt03",    20, 0., 10.);
  MuIso_nJets03_   = dbe_->book1D("MuIso_nJets03",   "Muon nJets03",   10, 0., 10.);
  MuIso_nTracks03_ = dbe_->book1D("MuIso_nTracks03", "Muon nTracks03", 20, 0., 20.);
  MuIso_sumPt03_   = dbe_->book1D("MuIso_sumPt03",   "Muon sumPt03",   20, 0., 40.);

  ElecIso_cal_ = dbe_->book1D("ElecIso_cal", "Electron Iso_cal", 50, -5., 45.);
  ElecIso_trk_ = dbe_->book1D("ElecIso_trk", "Electron Iso_trk", 50, -1.,  9.);

  // define logarithmic bins for a histogram with 100 bins going from 10^0 to 10^3

  const int nbins = 100;

  double logmin = 0.;
  double logmax = 3.;

  float bins[nbins+1];

  for (int i = 0; i <= nbins; i++) {

    double log = logmin + (logmax-logmin)*i/nbins;
    bins[i] = std::pow(10.0, log);

  }

  dimassRC_LOG_ = dbe_->book1D("dimassRC_LOG", "dimassRC_LOG", nbins, &bins[0]);
  dimassWC_LOG_ = dbe_->book1D("dimassWC_LOG", "dimassWC_LOG", nbins, &bins[0]);
  dimassRC_     = dbe_->book1D("dimassRC",     "dimassRC",     nbins, 0., 1000.);
  dimassWC_     = dbe_->book1D("dimassWC",     "dimassWC",     nbins, 0., 1000.);

  D_eta_muons_  = dbe_->book1D("D_eta_muons",  "#Delta eta_muons", 100, -5., 5.);
  D_phi_muons_  = dbe_->book1D("D_phi_muons",  "#Delta phi_muons", 100, -5., 5.);

  isoDimassCorrelation_ = dbe_->book2D("isoDimassCorrelation", "isoDimassCorrelation", 10, 0., 200., 10, 0., 1.);

  absCount_    = dbe_->book1D("absCount",    "absCount",    100, 0., 50.);
  relCount_    = dbe_->book1D("relCount",    "relCount",    100, 0.,  5.);
  combCount_   = dbe_->book1D("combCount",   "combCount",   100, 0.,  1.);
  diCombCount_ = dbe_->book1D("diCombCount", "diCombCount", 100, 0.,  2.);

}


void TopDiLeptonDQM::beginRun(const edm::Run& r, const EventSetup& context) {

}


void TopDiLeptonDQM::analyze(const edm::Event& evt, const edm::EventSetup& context) {

  edm::Handle<reco::MuonCollection> muons;
  evt.getByLabel(muons_, muons);

  if( muons.failedToGet() ) {

//    cout << endl << "------------------------" << endl;
//    cout << "--- NO RECO MUONS !! ---" << endl;
//    cout << "------------------------" << endl << endl;

    //    return;

  }

  edm::Handle<reco::GsfElectronCollection> elecs;
  evt.getByLabel(elecs_, elecs);

  if( elecs.failedToGet() ) {

//    cout << endl << "----------------------------" << endl;
//    cout << "--- NO RECO ELECTRONS !! ---" << endl;
//    cout << "----------------------------" << endl << endl;

    //    return;

  }

  edm::Handle<TriggerResults> trigResults;
  evt.getByLabel(triggerResults_, trigResults);

  if( trigResults.failedToGet() ) {

//    cout << endl << "-----------------------------" << endl;
//    cout << "--- NO TRIGGER RESULTS !! ---" << endl;
//    cout << "-----------------------------" << endl << endl;

  }

  const int n_TrigPaths = hltPaths_L3_.size();

  bool FiredTriggers[100] = {false};
  bool trigFired          =  false;

  if( !trigResults.failedToGet() ) {

    int n_Triggers = trigResults->size();

    TriggerNames trigName;
    trigName.init(*trigResults);

    for( int i_Trig = 0; i_Trig < n_Triggers; ++i_Trig ) {

      if (trigResults.product()->accept(i_Trig)) {

	for( int i = 0; i < n_TrigPaths; i++ ) {

	  if ( trigName.triggerName(i_Trig)== hltPaths_L3_[i] ) {

	    FiredTriggers[i] = true;
	    Trigs_->Fill(i);

	    //	  cout << endl << "-----------------------------" << endl;
	    //	    cout << "Trigger: " << hltPaths_L3_[i] << " FIRED!!!  " << endl;
	    //	    cout << "-----------------------------" << endl << endl;

	    trigFired = true;

	  }

	}

      }

    }

  }

  if( !muons.failedToGet() ) {

    Nmuons_->Fill( muons->size() );

    reco::MuonCollection::const_iterator muon;

    for(muon = muons->begin(); muon!= muons->end(); ++muon) {

      //      cout << " All muons p_T: " << muon->pt() << endl;
      //    cout << " All muons eta: " << muon->eta() << endl;
      //    cout << " All muons phi: " << muon->phi() << endl << endl;

      if(     muon->pt()   < muon_pT_cut_  )  continue;
      if( abs(muon->eta()) > muon_eta_cut_ )  continue;

      pT_muons_->Fill(  muon->pt() );
      eta_muons_->Fill( muon->eta() );
      phi_muons_->Fill( muon->phi() );

      reco::MuonIsolation muIso03 = muon->isolationR03();

      //      cout << " All muons sumPt: " << muIso03.sumPt << endl;
      //      cout << " All muons emEt:  " << muIso03.emEt  << endl;

      MuIso_emEt03_->Fill(    muIso03.emEt );
      MuIso_hadEt03_->Fill(   muIso03.hadEt );
      MuIso_hoEt03_->Fill(    muIso03.hoEt );
      MuIso_nJets03_->Fill(   muIso03.nJets );
      MuIso_nTracks03_->Fill( muIso03.nTracks );
      MuIso_sumPt03_->Fill(   muIso03.sumPt );

    }

    if( muons->size() > 2 ) {

      reco::MuonCollection::const_reference mu1 = muons->at(0);
      reco::MuonCollection::const_reference mu2 = muons->at(1);

      //  cout << " Mu1 p_T: " << mu1.pt()  << endl;
      //  cout << " Mu2 p_T: " << mu2.pt()  << endl;

      if( mu1.pt() > muon_pT_cut_ && abs(mu1.eta()) < muon_eta_cut_ ) {

	if( mu2.pt() > muon_pT_cut_ && abs(mu2.eta()) < muon_eta_cut_ ) {

	  //	  cout << endl << "-----------------------" << endl;
	  //	  cout << " Mu1 p_T: " << mu1.pt()  << endl;
	  //	  cout << " Mu2 p_T: " << mu2.pt()  << endl;

	  D_eta_muons_->Fill(mu1.eta()-mu2.eta());
	  D_phi_muons_->Fill(mu1.phi()-mu2.phi());

	  double dilepMass = sqrt( (mu1.energy()+mu2.energy())*(mu1.energy()+mu2.energy())
				   - (mu1.px()+mu2.px())*(mu1.px()+mu2.px())
				   - (mu1.py()+mu2.py())*(mu1.py()+mu2.py())
				   - (mu1.pz()+mu2.pz())*(mu1.pz()+mu2.pz())
				   );

	  //  cout << "--------------------" << endl;
	  //  cout << " Dimuon mass: " << dilepMass << endl;
	  //  cout << "--------------------" << endl << endl;

	  if( mu1.charge()*mu2.charge() < 0. ) {

	    dimassRC_LOG_->Fill( dilepMass );
	    dimassRC_->Fill( dilepMass );

	  }

	  if( mu1.charge()*mu2.charge() > 0. ) {

	    dimassWC_LOG_->Fill( dilepMass );
	    dimassWC_->Fill( dilepMass );

	  }

	}

      }

    }

  }


  if( !elecs.failedToGet() ) {

    Nelecs_->Fill( elecs->size() );

    reco::GsfElectronCollection::const_iterator elec;

    //    cout << endl  << "--------------------" << endl;

    for(elec = elecs->begin(); elec!= elecs->end(); ++elec) {

      //      cout << " All electrons p_T: " << elec->pt() << endl;
      //    cout << " All electrons eta: " << elec->eta() << endl;
      //    cout << " All electrons phi: " << elec->phi() << endl << endl;

      if(     elec->pt()   < elec_pT_cut_  )  continue;
      if( abs(elec->eta()) > elec_eta_cut_ )  continue;

      pT_elecs_->Fill( elec->pt() );
      eta_elecs_->Fill(elec->eta());
      phi_elecs_->Fill(elec->phi());

      reco::GsfElectron::IsolationVariables elecIso = elec->dr03IsolationVariables();

      //      cout << " All electrons EcalSumEt: " << elecIso.ecalRecHitSumEt << endl;
      //      cout << " All electrons tkSumPt:   " << elecIso.tkSumPt << endl;

      ElecIso_cal_->Fill( elecIso.ecalRecHitSumEt );
      ElecIso_trk_->Fill( elecIso.tkSumPt );

    }

  }

}


void TopDiLeptonDQM::endRun(const Run& r, const EventSetup& context) {

}

void TopDiLeptonDQM::endJob() {

}
