/*
 *  $Date: 2009/07/13 10:11:07 $
 *  $Revision: 1.1 $
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
  muons_      = ps.getParameter<edm::InputTag>("muonCollection");
  pT_cut_     = ps.getParameter<double>("pT_cut");
  eta_cut_    = ps.getParameter<double>("eta_cut");
  moduleName_ = ps.getUntrackedParameter<string>("moduleName");

}


TopDiLeptonDQM::~TopDiLeptonDQM() {

}


void TopDiLeptonDQM::initialize() {

}


void TopDiLeptonDQM::beginJob(const edm::EventSetup& evt) {

  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder(moduleName_);

  Nmuons_    = dbe_->book1D("Nmuons",    "Nmuons",     10,  0.,  10.);
  pT_muons_  = dbe_->book1D("pT_muons",  "pT_muons",  100,  0., 200.);
  eta_muons_ = dbe_->book1D("eta_muons", "eta_muons", 100, -5.,   5.);
  phi_muons_ = dbe_->book1D("phi_muons", "phi_muons",  80, -4.,   4.);

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

    cout << endl << "------------------------" << endl;
    cout << "--- NO RECO MUONS !! ---" << endl;
    cout << "------------------------" << endl << endl;

    return;

  }

  reco::MuonCollection::const_iterator muon;

  for(muon = muons->begin(); muon!= muons->end(); ++muon) {

    //    cout << " p_T: " << muon->pt()  << endl;
    //    cout << " eta: " << muon->eta() << endl;
    //    cout << " phi: " << muon->phi() << endl << endl;

    if(     muon->pt()   < pT_cut_  )  continue;
    if( abs(muon->eta()) > eta_cut_ )  continue;

    pT_muons_->Fill( muon->pt() );
    eta_muons_->Fill(muon->eta());
    phi_muons_->Fill(muon->phi());

  }

  Nmuons_->Fill( muons->size() );

  if( muons->size() < 2 )  return;

  reco::MuonCollection::const_reference mu1 = muons->at(0);
  reco::MuonCollection::const_reference mu2 = muons->at(1);

  if( mu1.pt() < pT_cut_ || abs(mu1.eta()) > eta_cut_ )  return;
  if( mu2.pt() < pT_cut_ || abs(mu2.eta()) > eta_cut_ )  return;

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

  reco::MuonIsolation muIso03_1 = mu1.isolationR03();
  reco::MuonIsolation muIso03_2 = mu2.isolationR03();

  //  emEt03_   ->Fill( muIso03.emEt,   weight );
  //  hadEt03_  ->Fill( muIso03.hadEt,  weight );
  //  hoEt03_   ->Fill( muIso03.hoEt,   weight ); 
  //  nTracks03_->Fill( muIso03.nTracks,weight );
  //  sumPt03_  ->Fill( muIso03.sumPt,  weight );

  //  double absTrackIso1 = mu1.trackIso();
  //  double absTrackIso2 = mu2.trackIso();

  //  double absCaloIso1 = mu1.caloIso();
  //  double absCaloIso2 = mu2.caloIso();  

  //  double relTrackIso1 = mu1.trackIso()/mu1.pt();
  //  double relTrackIso2 = mu2.trackIso()/mu2.pt();

  //  double relCaloIso1 = mu1.caloIso()/mu1.pt();
  //  double relCaloIso2 = mu2.caloIso()/mu2.pt();  

  //  double combIso1 = mu1.pt()/(mu1.pt()+mu1.trackIso()+mu1.caloIso());
  //  double combIso2 = mu2.pt()/(mu2.pt()+mu2.trackIso()+mu2.caloIso());     

  //  double diCombIso = sqrt(combIso1*combIso1+combIso2*combIso2);

  //  isoDimassCorrelation_->Fill( dilepMass, combIso1 );
  //  isoDimassCorrelation_->Fill( dilepMass, combIso2 );

  //  for( int i = 1; i <= 100; ++i ) {

  //    if( diCombIso>0.02*i ) {
  //      diCombCount_->SetBinContent(i,diCombCount_->GetBinContent(i)+1);
  //    }

  //    if( combIso1>0.01*i && combIso2>0.01*i ) {
  //      combCount_->SetBinContent(i,combCount_->GetBinContent(i)+1);
  //    }

  //    if( relTrackIso1<(0.05*i) && relCaloIso1<(0.05*i) && relTrackIso2<(0.05*i) && relCaloIso2<(0.05*i) ) {
  //      relCount_->SetBinContent(i,relCount_->GetBinContent(i)+1);
  //    }

  //    if( absTrackIso1<(0.5*i) && absCaloIso1<(0.5*i) && absTrackIso2<(0.5*i) && absCaloIso2<(0.5*i) ) {
  //      absCount_->SetBinContent(i,absCount_->GetBinContent(i)+1);
  //    }

  //  }

}


void TopDiLeptonDQM::endRun(const Run& r, const EventSetup& context) {

}

void TopDiLeptonDQM::endJob() {

}
