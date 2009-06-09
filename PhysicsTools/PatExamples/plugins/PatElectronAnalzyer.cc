#include "TH1.h"
#include "TH2.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

class PatElectronAnalyzer : public edm::EDAnalyzer {

 public:
  
  explicit PatElectronAnalyzer(const edm::ParameterSet&);
  ~PatElectronAnalyzer();
  
 private:
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // decide in which mode to run the analyzer
  // 0:plain, 1:genMatch, 2:tagAndProbe are 
  // supported depending on the line comments
  unsigned int mode_;

  // source of electrons
  edm::InputTag electronSrc_;
  // source of generator particles
  edm::InputTag particleSrc_;

  edm::ParameterSet genMatchMode_;
  edm::ParameterSet tagAndProbeMode_;

  // internal variables for genMatchMode and
  // tagAndProbeMode 
  double maxDeltaR_;
  double maxDeltaM_;
  double maxTagIso_;
  
  // book histograms of interest
  TH1I *nr_;
  TH1F *pt_;
  TH1F *eta_;
  TH1F *phi_;
  TH1F *genPt_;
  TH1F *genEta_;
  TH1F *genPhi_;
  TH1F *deltaR_;
  TH1F *isoTag_;
  TH1F *invMass_;
  TH1F *deltaPhi_;
};  

#include "Math/VectorUtil.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

PatElectronAnalyzer::PatElectronAnalyzer(const edm::ParameterSet& cfg):
  mode_ (cfg.getParameter<unsigned int>("mode")),
  electronSrc_ (cfg.getParameter<edm::InputTag>("electronSrc")),
  particleSrc_ (cfg.getParameter<edm::InputTag>("particleSrc")),
  genMatchMode_(cfg.getParameter<edm::ParameterSet>("genMatchMode")),
  tagAndProbeMode_(cfg.getParameter<edm::ParameterSet>("tagAndProbeMode"))
{
  // complete the configuration of the analyzer
  maxDeltaR_ = genMatchMode_   .getParameter<double>("maxDeltaR"); 
  maxDeltaM_ = tagAndProbeMode_.getParameter<double>("maxDeltaM"); 
  maxTagIso_ = tagAndProbeMode_.getParameter<double>("maxTagIso"); 

  // register histograms to the TFileService
  edm::Service<TFileService> fs;
  
  nr_      = fs->make<TH1I>("nr",       "nr(e)",     10,   0 ,  10 );
  pt_      = fs->make<TH1F>("pt",       "p(e)",      20,   0., 100.);
  eta_     = fs->make<TH1F>("eta",      "eta(e)",    30,  -3.,   3.);
  phi_     = fs->make<TH1F>("phi",      "phi",       35, -3.5,  3.5);
  genPt_   = fs->make<TH1F>("genPt",    "p(e)",      20,   0., 100.);
  genEta_  = fs->make<TH1F>("genEta",   "eta(e)",    30,  -3.,   3.);
  genPhi_  = fs->make<TH1F>("genPhi",   "phi",       35, -3.5,  3.5);
  deltaR_  = fs->make<TH1F>("deltaR",   "deltaR",    50,   0.,   5.);
  isoTag_  = fs->make<TH1F>("isoTag",   "iso(e)",    50,   0.,  10.);
  invMass_ = fs->make<TH1F>("invMass",  "m(ee)",    100,  50., 150.);
  deltaPhi_= fs->make<TH1F>("deltaPhi", "deltaPhi", 100, -3.5,  3.5);
}

PatElectronAnalyzer::~PatElectronAnalyzer()
{
}

void
PatElectronAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{       
  // get electron collection
  edm::Handle<std::vector<pat::Electron> > electrons;
  evt.getByLabel(electronSrc_, electrons); 
  // get generator particle collection
  edm::Handle<reco::GenParticleCollection> particles;
  evt.getByLabel(particleSrc_, particles); 


  for(reco::GenParticleCollection::const_iterator part=particles->begin(); 
      part!=particles->end(); ++part){
    // only loop stable electrons
    if( part->status()==1  && abs(part->pdgId())==11 ){
      genPt_ ->Fill( part->pt()  );
      genEta_->Fill( part->eta() );
      genPhi_->Fill( part->phi() );      
    }
  }

  nr_->Fill( electrons->size() );
  for( std::vector<pat::Electron>::const_iterator elec=electrons->begin();
       elec!=electrons->end(); ++elec){
    //
    // uncomment the following lines to enable >> Mode 0: plain
    //
    if( mode_==0 ){
      // uncomment the following line and choose a type
      // to require a certain electronID; the following 
      // types are available:
      // * eidRobustLoose
      // * eidRobustTight
      // * eidLoose
      // * eidTight
      // * eidRobustHighEnergy
      //if( !elec->electronID("dummyName")>0.5 ) continue;
      pt_ ->Fill( elec->pt()  );
      eta_->Fill( elec->eta() );
      phi_->Fill( elec->phi() );
    }

    //
    // uncomment the following lines to enable >> Mode 1: genMatch
    //
    if( mode_==1 ){
      if( elec->genLepton() ){
	float deltaR = ROOT::Math::VectorUtil::DeltaR(elec->genLepton()->p4(), elec->p4());
	deltaR_->Fill(deltaR);	
	if( deltaR<maxDeltaR_ ){
	  // uncomment the following line and choose a type
	  // to require a certain electronID; the following 
	  // types are available:
	  // * eidRobustLoose
	  // * eidRobustTight
	  // * eidLoose
	  // * eidTight
	  // * eidRobustHighEnergy
	  //if( !elec->electronID("dummyName")>0.5 ) continue;
	  pt_ ->Fill( elec->pt()  );
	  eta_->Fill( elec->eta() );
	  phi_->Fill( elec->phi() );
	}
      }

    //
    // uncomment the following lines to enable >> Mode 2: tagAndProbe
    //
      if( mode_==2 ){
	isoTag_->Fill(elec->trackIso());	
	if( elec->trackIso()<maxTagIso_  && elec->electronID("eidTight")>0.5 ){
	  for( std::vector<pat::Electron>::const_iterator probe=electrons->begin();
	       probe!=electrons->end(); ++probe){
	    // skip the tagged electron itself
	    if( probe==elec ) continue;

	    float zMass = (probe->p4()+elec->p4()).mass();
	    invMass_ ->Fill(zMass);	
	    float deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(elec->p4(), probe->p4());
	    deltaPhi_->Fill(deltaPhi);	
	    
	    // check for the Z mass
	    if( fabs( zMass-90. )<maxDeltaM_ ){
	      // uncomment the following line and choose a type
	      // to require a certain electronID; the following 
	      // types are available:
	      // * eidRobustLoose
	      // * eidRobustTight
	      // * eidLoose
	      // * eidTight
	      // * eidRobustHighEnergy
	      //if( !elec->electronID("dummyName")>0.5 ) continue;
	      pt_ ->Fill( elec->pt()  );
	      eta_->Fill( elec->eta() );
	      phi_->Fill( elec->phi() );
	    }
	  }
	}
      }
    }
  }
}

void PatElectronAnalyzer::beginJob(const edm::EventSetup&)
{
}

void PatElectronAnalyzer::endJob()
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatElectronAnalyzer);
