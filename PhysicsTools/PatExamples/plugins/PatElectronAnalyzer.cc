#include "TH1.h"
#include "TH2.h"
#include "TMath.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class PatElectronAnalyzer : public edm::EDAnalyzer {

 public:

  explicit PatElectronAnalyzer(const edm::ParameterSet&);
  ~PatElectronAnalyzer();

 private:

  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  // restrictions for the electron to be
  // considered
  double minPt_;
  double maxEta_;

  // decide in which mode to run the analyzer
  // 0 : genMatch, 1 : tagAndProbe are
  // supported depending on the line comments
  unsigned int mode_;

  // choose a given electronID for the electron
  // in consideration; the following types are
  // available:
  // * eidRobustLoose
  // * eidRobustTight
  // * eidLoose
  // * eidTight
  // * eidRobustHighEnergy
  std::string electronID_;

  // source of electrons
  edm::EDGetTokenT<std::vector<pat::Electron> > electronSrcToken_;
  // source of generator particles
  edm::EDGetTokenT<reco::GenParticleCollection> particleSrcToken_;

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

PatElectronAnalyzer::PatElectronAnalyzer(const edm::ParameterSet& cfg):
  minPt_ (cfg.getParameter<double>("minPt")),
  maxEta_ (cfg.getParameter<double>("maxEta")),
  mode_ (cfg.getParameter<unsigned int>("mode")),
  electronID_  (cfg.getParameter<std::string>("electronID")),
  electronSrcToken_ (consumes<std::vector<pat::Electron> >(cfg.getParameter<edm::InputTag>("electronSrc"))),
  particleSrcToken_ (consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("particleSrc"))),
  genMatchMode_(cfg.getParameter<edm::ParameterSet>("genMatchMode")),
  tagAndProbeMode_(cfg.getParameter<edm::ParameterSet>("tagAndProbeMode"))
{
  // complete the configuration of the analyzer
  maxDeltaR_ = genMatchMode_   .getParameter<double>("maxDeltaR");
  maxDeltaM_ = tagAndProbeMode_.getParameter<double>("maxDeltaM");
  maxTagIso_ = tagAndProbeMode_.getParameter<double>("maxTagIso");


  // register histograms to the TFileService
  edm::Service<TFileService> fs;
  nr_      = fs->make<TH1I>("nr",       "nr",        10,   0 ,  10 );
  pt_      = fs->make<TH1F>("pt",       "pt",        20,   0., 100.);
  eta_     = fs->make<TH1F>("eta",      "eta",       30,  -3.,   3.);
  phi_     = fs->make<TH1F>("phi",      "phi",       35, -3.5,  3.5);
  genPt_   = fs->make<TH1F>("genPt",    "pt",        20,   0., 100.);
  genEta_  = fs->make<TH1F>("genEta",   "eta",       30,  -3.,   3.);
  genPhi_  = fs->make<TH1F>("genPhi",   "phi",       35, -3.5,  3.5);
  deltaR_  = fs->make<TH1F>("deltaR",   "log(dR)",   50,  -5.,   0.);
  isoTag_  = fs->make<TH1F>("isoTag",   "iso",       50,   0.,  10.);
  invMass_ = fs->make<TH1F>("invMass",  "m",        100,  50., 150.);
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
  evt.getByToken(electronSrcToken_, electrons);
  // get generator particle collection
  edm::Handle<reco::GenParticleCollection> particles;
  evt.getByToken(particleSrcToken_, particles);

  nr_->Fill( electrons->size() );

  // ----------------------------------------------------------------------
  //
  // First Part Mode 0: genMatch
  //
  // ----------------------------------------------------------------------
  if( mode_==0 ){
    // loop generator particles
    for(reco::GenParticleCollection::const_iterator part=particles->begin();
	part!=particles->end(); ++part){
      // only loop stable electrons
      if( part->status()==1  && abs(part->pdgId())==11 ){
	if( part->pt()>minPt_ && fabs(part->eta())<maxEta_ ){
	  genPt_ ->Fill( part->pt()  );
	  genEta_->Fill( part->eta() );
	  genPhi_->Fill( part->phi() );
	}
      }
    }

    // loop electrons
    for( std::vector<pat::Electron>::const_iterator elec=electrons->begin(); elec!=electrons->end(); ++elec ){
      if( elec->genLepton() ){
	float deltaR = ROOT::Math::VectorUtil::DeltaR(elec->genLepton()->p4(), elec->p4());
	deltaR_->Fill(TMath::Log10(deltaR));
	if( deltaR<maxDeltaR_ ){
	  if( electronID_!="none" ){
	    if( elec->electronID(electronID_)<0.5 )
	      continue;
	  }
	  if( elec->pt()>minPt_ && fabs(elec->eta())<maxEta_ ){
	    pt_ ->Fill( elec->pt()  );
	    eta_->Fill( elec->eta() );
	    phi_->Fill( elec->phi() );
	  }
	}
      }
    }
  }

  // ----------------------------------------------------------------------
  //
  // Second Part Mode 1: tagAndProbe
  //
  // ----------------------------------------------------------------------
  if( mode_==1 ){
    // loop tag electron
    for( std::vector<pat::Electron>::const_iterator elec=electrons->begin(); elec!=electrons->end(); ++elec ){
      isoTag_->Fill(elec->trackIso());
      if( elec->trackIso()<maxTagIso_  && elec->electronID("eidTight")>0.5 ){
	// loop probe electron
	for( std::vector<pat::Electron>::const_iterator probe=electrons->begin(); probe!=electrons->end(); ++probe ){
	  // skip the tag electron itself
	  if( probe==elec ) continue;

	  float zMass = (probe->p4()+elec->p4()).mass();
	  invMass_ ->Fill(zMass);
	  float deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(elec->p4(), probe->p4());
	  deltaPhi_->Fill(deltaPhi);

	  // check for the Z mass
	  if( fabs( zMass-90. )<maxDeltaM_ ){
	    if( electronID_!="none" ){
	      if( probe->electronID(electronID_)<0.5 )
		continue;
	    }
	    if( probe->pt()>minPt_ && fabs(probe->eta())<maxEta_ ){
	      pt_ ->Fill( probe->pt()  );
	      eta_->Fill( probe->eta() );
	      phi_->Fill( probe->phi() );
	    }
	  }
	}
      }
    }
  }
}

void PatElectronAnalyzer::beginJob()
{
}

void PatElectronAnalyzer::endJob()
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatElectronAnalyzer);
