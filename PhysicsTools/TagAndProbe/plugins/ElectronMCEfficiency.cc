#ifndef PhysicsTools_TagAndProbe_ElectronMCEfficiency
#define PhysicsTools_TagAndProbe_ElectronMCEfficiency

#include <string>
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h" // reco::deltaR
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TTree.h"

/* ******************************************************** */
/**  Store Monte Carlo generated  information to compute efficiency */
/* ******************************************************** */


class ElectronMCEfficiency : public edm::EDAnalyzer
{
 public:
  explicit ElectronMCEfficiency (const edm::ParameterSet&);
  ~ElectronMCEfficiency();

 private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual bool boolResults(  const edm::Event&, const reco::Candidate&);  
  virtual bool CheckAcceptance( const edm::Event&, 
			       const reco::Candidate&);

  bool isFromResDecay(int);

  bool overlap( const reco::Candidate &, 
			      const reco::Candidate &) const;

  bool CheckSuperClusterMatch( const edm::Event&, const reco::Candidate&);
  void setDef(void);

  // ----------member data ---------------------------

  TFile*      hOutputFile ;
  TTree *     myTree;
  std::string fOutputFileName ;


  edm::InputTag  SuperClusters_;
  std::vector<int> truthParentId_;
  double ElectronPtCut_;
  double dRWindow_;

  float mRes;
  float Z_px;
  float Z_py;
  float Z_pz;
  float Z_E;
  float Z_Pt;
  float Z_Et;
  float Z_Eta;   
  float Z_Phi;
  float Z_Vx;
  float Z_Vy;
  float Z_Vz;
     
  float e1_px;
  float e1_py;
  float e1_pz;
  float e1_E;
  float e1_Pt;
  float e1_Et;
  float e1_Eta;    
  float e1_Phi;
  int e1_Charge;
  float e1_Vx;
  float e1_Vy;
  float e1_Vz;
  bool e1_isWP80;

  float e2_px;
  float e2_py;
  float e2_pz;
  float e2_E;
  float e2_Pt;
  float e2_Et;
  float e2_Eta;    
  float e2_Phi;
  int e2_Charge;
  float e2_Vx;
  float e2_Vy;
  float e2_Vz;
  bool e2_isWP80;

};
#endif






ElectronMCEfficiency::ElectronMCEfficiency(const edm::ParameterSet& iConfig)
{
  SuperClusters_      = iConfig.getParameter<edm::InputTag>("SuperClusters");


  // MC truth parent Id
  std::vector<int>       dEmptyIntVec;
  truthParentId_  = iConfig.getUntrackedParameter< std::vector<int> >("MCTruthParentId", 
								      dEmptyIntVec);

  // pT cut
  ElectronPtCut_ = 25.0;


  // deltaR matching window
  dRWindow_ = 0.3;

  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<std::string>("HistOutFile",
							       "demo.root"); 
}





ElectronMCEfficiency::~ElectronMCEfficiency()
{
  // Clean up
}





  void ElectronMCEfficiency::beginJob()
  {
    // Open output ROOT file and TTree
    hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ; 
    myTree = new TTree("probe","Probe MC Truth Tree");

    myTree->Branch("mRes",        &mRes,        "mRes/F");
    myTree->Branch("Z_px",      &Z_px,      "Z_px/F");
    myTree->Branch("Z_py",      &Z_py,      "Z_py/F");
    myTree->Branch("Z_pz",      &Z_pz,      "Z_pz/F");
    myTree->Branch("Z_E",       &Z_E,       "Z_E/F");
    myTree->Branch("Z_Pt",      &Z_Pt,      "Z_Pt/F");
    myTree->Branch("Z_Et",      &Z_Et,      "Z_Et/F");
    myTree->Branch("Z_Eta",     &Z_Eta,     "Z_Eta/F");    
    myTree->Branch("Z_Phi",     &Z_Phi,     "Z_Phi/F");
    myTree->Branch("Z_Vx",      &Z_Vx,      "Z_Vx/F");
    myTree->Branch("Z_Vy",      &Z_Vy,      "Z_Vy/F");
    myTree->Branch("Z_Vz",      &Z_Vz,      "Z_Vz/F");

    myTree->Branch("e1_px",     &e1_px,     "e1_px/F");
    myTree->Branch("e1_py",     &e1_py,     "e1_py/F");
    myTree->Branch("e1_pz",     &e1_pz,     "e1_pz/F");
    myTree->Branch("e1_E",      &e1_E,      "e1_E/F");
    myTree->Branch("e1_Pt",     &e1_Pt,     "e1_Pt/F");
    myTree->Branch("e1_Et",     &e1_Et,     "e1_Et/F");    
    myTree->Branch("e1_Eta",    &e1_Eta,    "e1_Eta/F");    
    myTree->Branch("e1_Phi",    &e1_Phi,    "e1_Phi/F");
    myTree->Branch("e1_Charge", &e1_Charge, "e1_Charge/I");
    myTree->Branch("e1_Vx",     &e1_Vx,     "e1_Vx/F");
    myTree->Branch("e1_Vy",     &e1_Vy,     "e1_Vy/F");
    myTree->Branch("e1_Vz",     &e1_Vz,     "e1_Vz/F");
    myTree->Branch("e1_isWP80", &e1_isWP80, "e1_isWP80/O");

    myTree->Branch("e2_px",     &e2_px,     "e2_px/F");
    myTree->Branch("e2_py",     &e2_py,     "e2_py/F");
    myTree->Branch("e2_pz",     &e2_pz,     "e2_pz/F");
    myTree->Branch("e2_E",      &e2_E,      "e2_E/F");
    myTree->Branch("e2_Pt",     &e2_Pt,     "e2_Pt/F");
    myTree->Branch("e2_Et",     &e2_Et,     "e2_Et/F");    
    myTree->Branch("e2_Eta",    &e2_Eta,    "e2_Eta/F");    
    myTree->Branch("e2_Phi",    &e2_Phi,    "e2_Phi/F");
    myTree->Branch("e2_Charge", &e2_Charge, "e2_Charge/I");
    myTree->Branch("e2_Vx",     &e2_Vx,     "e2_Vx/F");
    myTree->Branch("e2_Vy",     &e2_Vy,     "e2_Vy/F");
    myTree->Branch("e2_Vz",     &e2_Vz,     "e2_Vz/F");
    myTree->Branch("e2_isWP80", &e2_isWP80, "e2_isWP80/O");
  }




  void ElectronMCEfficiency::endJob()
  {
    hOutputFile->SetCompressionLevel(2);
     hOutputFile->cd();
     myTree->Write();

    delete myTree;
    hOutputFile->Close();
    delete hOutputFile;
  }



void ElectronMCEfficiency::setDef(void){

  mRes               = -1.;
  Z_px             = -99999.;
  Z_py             = -99999.;
  Z_pz             = -99999.;
  Z_E              = -1.;
  Z_Pt             = -1.;
  Z_Et             = -1.;
  Z_Eta            = -10.;
  Z_Phi            = -10.;
  Z_Vx             = -10.;
  Z_Vy             = -10.;
  Z_Vz             = -10.;

  e1_px            = -99999.;
  e1_py            = -99999.;
  e1_pz            = -99999.;
  e1_E             = -1.;
  e1_Pt            = -1.;
  e1_Et            = -1.;
  e1_Eta           = -10.;
  e1_Phi           = -10.;
  e1_Charge        = -10;
  e1_Vx            = -10.;
  e1_Vy            = -10.;
  e1_Vz            = -10.;   
  e1_isWP80        = false;

  e2_px            = -99999.;
  e2_py            = -99999.;
  e2_pz            = -99999.;
  e2_E             = -1.;
  e2_Pt            = -1.;
  e2_Et            = -1.;
  e2_Eta           = -10.;
  e2_Phi           = -10.;
  e2_Charge        = -10;
  e2_Vx            = -10.;
  e2_Vy            = -10.;
  e2_Vz            = -10.;   
  e2_isWP80        = false;
}



void ElectronMCEfficiency::analyze(const edm::Event& iEvent, 
				    const edm::EventSetup& iSetup) {

  setDef();

  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByLabel("genParticles", genParticles);

  size_t nZ = genParticles->size();
  if( nZ < 1 ) return;
  const reco::Candidate *Res(0);

  for(size_t i = 0; i < nZ; ++ i) {

    Res = &((*genParticles)[i]);
    size_t ndau = 0;
    if(!(Res==0)) ndau = Res->numberOfDaughters();
    if( ndau<2 || !isFromResDecay(Res->pdgId()) ) continue;
    const reco::Candidate *d1 = Res->daughter( 0 );
    const reco::Candidate *d2 = Res->daughter( 1 );

    if( d1==0 || d2==0 ) continue;
    if( ! (abs(d1->pdgId()) == 11 && abs(d2->pdgId()) == 11) ) continue;
    bool isInAcceptance = CheckAcceptance(iEvent, *d1);
    bool isSuperCluster = CheckSuperClusterMatch( iEvent, *d1);
    if(!(isInAcceptance && isSuperCluster)) continue;
    isInAcceptance = CheckAcceptance(iEvent, *d2);
    isSuperCluster = CheckSuperClusterMatch( iEvent, *d2);
    if(!(isInAcceptance && isSuperCluster)) continue;


    // setDef();

    mRes    = Res->mass();
    Z_px  = Res->px();
    Z_py  = Res->py();
    Z_pz  = Res->pz();
    Z_E   = Res->energy();
    Z_Pt  = Res->pt();
    Z_Et  = Res->et();
    Z_Eta = Res->eta();   
    Z_Phi = Res->phi();
    Z_Vx  = Res->vx();
    Z_Vy  = Res->vy();
    Z_Vz  = Res->vz();
    
    e1_px     = d1->px();
    e1_py     = d1->py();
    e1_pz     = d1->pz();
    e1_E      = d1->energy();
    e1_Pt     = d1->pt();
    e1_Et     = d1->et();
    e1_Eta    = d1->eta();    
    e1_Phi    = d1->phi();
    e1_Charge = d1->charge();
    e1_Vx     = d1->vx();
    e1_Vy     = d1->vy();
    e1_Vz     = d1->vz();
    e1_isWP80 = boolResults( iEvent, *d1);

  
    e2_px     = d2->px();
    e2_py     = d2->py();
    e2_pz     = d2->pz();
    e2_E      = d2->energy();
    e2_Pt     = d2->pt();
    e2_Et     = d2->et();
    e2_Eta    = d2->eta();    
    e2_Phi    = d2->phi();
    e2_Charge = d2->charge();
    e2_Vx     = d2->vx();
    e2_Vy     = d2->vy();
    e2_Vz     = d2->vz();
    e2_isWP80 = boolResults( iEvent, *d2);

      // fill the tree for each candidate
      myTree->Fill();
  }

}
// ---------------------------------








bool ElectronMCEfficiency::boolResults( const edm::Event& iEvent, 
					const reco::Candidate& genele) {

   bool result = false;

   // ********* List all cuts here ******************* //
   // WP80 cuts
   const float MAX_MissingHits      = 0.0;
   const float MIN_Dist             = 0.02;
   const float MIN_Dcot             = 0.02;

   const float cut_EB_trackRel03    = 0.09;
   const float cut_EB_ecalRel03     = 0.07;
   const float cut_EB_hcalRel03     = 0.10;
   const float cut_EB_sigmaIetaIeta = 0.01;
   const float cut_EB_deltaPhi      = 0.06;
   const float cut_EB_deltaEta      = 0.004;
   const float cut_EB_HoverE        = 0.04;

   const float cut_EE_trackRel03    = 0.04;
   const float cut_EE_ecalRel03     = 0.05;
   const float cut_EE_hcalRel03     = 0.025;
   const float cut_EE_sigmaIetaIeta = 0.03;
   const float cut_EE_deltaPhi      = 0.03;
   const float cut_EE_deltaEta      = 0.007; 
   const float cut_EE_HoverE        = 0.025;



  // --------- Read the electron collection in the event
  edm::Handle<edm::View<reco::GsfElectron> > electrons;
  iEvent.getByLabel("gsfElectrons", electrons);

 

  // loop over electron collection
  for(edm::View<reco::GsfElectron>::const_iterator  
	elec = electrons->begin(); elec != electrons->end();++elec) {

     float deltaPhi = elec->deltaPhiSuperClusterTrackAtVtx();
     float deltaEta = elec->deltaEtaSuperClusterTrackAtVtx();
     float HoverE = elec->hcalOverEcal();
     float trackiso = elec->dr03TkSumPt() / elec->et();
     float ecaliso = elec->dr03EcalRecHitSumEt() / elec->et();
     float hcaliso = elec->dr03HcalTowerSumEt() / elec->et();
     float sigmaIetaIeta = elec->sigmaIetaIeta();
     float convDist = fabs(elec->convDist()); 
     float convDcot = fabs(elec->convDcot()); 
    
     bool select = true;
     if( !(elec->ecalDrivenSeed==1) )  select = false; 
     if(convDist<MIN_Dist && convDcot<MIN_Dcot) select = false;
     if(elec->gsfTrack()->trackerExpectedHitsInner().numberOfHits()>MAX_MissingHits) select = false;
     if( elec->isEB() ) {
        if( fabs(deltaPhi)>cut_EB_deltaPhi ) select = false;
        if( fabs(deltaEta)>cut_EB_deltaEta ) select = false;
        if( HoverE>cut_EB_HoverE ) select = false;
        if( trackiso>cut_EB_trackRel03 ) select = false;
        if( ecaliso>cut_EB_ecalRel03 ) select = false;
        if( hcaliso>cut_EB_hcalRel03 ) select = false;
        if( sigmaIetaIeta > cut_EB_sigmaIetaIeta ) select = false;
     }    
     else if( elec->isEE() ) {
        if( fabs(deltaPhi)>cut_EE_deltaPhi ) select = false;
        if( fabs(deltaEta)>cut_EE_deltaEta ) select = false;
        if( HoverE>cut_EE_HoverE ) select = false;
        if( trackiso>cut_EE_trackRel03 ) select = false;
        if( ecaliso>cut_EE_ecalRel03 ) select = false;
        if( hcaliso>cut_EE_hcalRel03 ) select = false;
        if( sigmaIetaIeta > cut_EE_sigmaIetaIeta ) select = false;
     }
     else select = false;

     if( select && overlap( *elec, genele) ) {
        result = true;
        break;
     }
     
  }
  return result;
}









////////// check the MC truth Id of the parent ///////////////////

bool ElectronMCEfficiency::isFromResDecay(int pdgId) {
  
  bool result = false;  
  for(int j=0; j< (int) truthParentId_.size(); j++) {
    if(pdgId == truthParentId_[j]) {
      result = true;
      break;
    }
  }

  return result;
}





////////// Apply event selection cuts ///////////////////

bool ElectronMCEfficiency::CheckAcceptance( const edm::Event& iEvent, 
					   const reco::Candidate& ele ) {

  bool result = true;

  float eEta = ele.eta();
  float ePt  = ele.pt();


  // electron pT cut
  if( ePt < ElectronPtCut_ ) result = false;

  // electron acceptance
  if( !((fabs(eEta)<1.4442) || 
	(fabs(eEta)>1.566 && fabs(eEta)<2.5)) ) result = false;

  return result;
}






//  ************* Utility: check overlap **********************
bool ElectronMCEfficiency::overlap( const reco::Candidate & e, 
				    const reco::Candidate & c) const {

  if( fabs(deltaR(e,c) ) < dRWindow_ ) return true;
    
  return false;
}
//--------------------------------------------










////////// Does the MC generated electron match a super cluster ///////////////////

bool ElectronMCEfficiency::CheckSuperClusterMatch( const edm::Event& iEvent, 
						   const reco::Candidate& ele ) {

  bool result = false;

  edm::Handle<edm::View<reco::Candidate> > SuperClusters;
  iEvent.getByLabel( SuperClusters_, SuperClusters);

  for(edm::View<reco::Candidate>::const_iterator  
	sc = SuperClusters->begin(); sc != SuperClusters->end();++sc) {
    if( overlap( *sc, ele) ) {
      result = true; 
      break;
    }
  }
  return result; 
}



//define this as a plug-in
DEFINE_FWK_MODULE( ElectronMCEfficiency );

