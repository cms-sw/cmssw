// -*- C++ -*-
//
// L1TSummary:  produce command line visible summary of L1T system
// 

#include <iostream>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

using namespace std;
using namespace edm;
using namespace l1t;

class L1TSummary : public EDAnalyzer {
public:
  explicit L1TSummary(const ParameterSet&);
  ~L1TSummary();
  
  static void fillDescriptions(ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginJob();
  virtual void analyze(Event const&, EventSetup const&);
  virtual void endJob();
  
  virtual void beginRun(Run const&, EventSetup const&);
  virtual void endRun(Run const&, EventSetup const&);
  virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&);
  virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&);

  // Tag string to mark summary with:
  string tag_;
      
  // Checks to perform:
  bool egCheck_;   
  bool tauCheck_;  
  bool jetCheck_;  
  bool sumCheck_;  
  bool muonCheck_; 

  // EDM tokens:
  edm::EDGetTokenT<EGammaBxCollection> egToken_;
  edm::EDGetTokenT<TauBxCollection> tauToken_;
  edm::EDGetTokenT<JetBxCollection> jetToken_;
  edm::EDGetTokenT<EtSumBxCollection> sumToken_;
  edm::EDGetTokenT<MuonBxCollection> muonToken_;

};

L1TSummary::L1TSummary(const ParameterSet& iConfig) { 


  // InputTag barrelTfInputTag = iConfig.getParameter<InputTag>("barrelTFInput");
  // InputTag overlapTfInputTag = iConfig.getParameter<InputTag>("overlapTFInput");
  // InputTag forwardTfInputTag = iConfig.getParameter<InputTag>("forwardTFInput");
  //m_barrelTfInputToken = consumes<MicroGMTConfiguration::InputCollection>(iConfig.getParameter<InputTag>("bmtfDigis"));

  tag_       = iConfig.getParameter<string>("tag");

  egCheck_   = iConfig.getParameter<bool>("egCheck");
  tauCheck_  = iConfig.getParameter<bool>("tauCheck");
  jetCheck_  = iConfig.getParameter<bool>("jetCheck");
  sumCheck_  = iConfig.getParameter<bool>("sumCheck");
  muonCheck_ = iConfig.getParameter<bool>("muonCheck");

  cout << "DEBUG:  egCheck:    " << egCheck_ << "\n";
  cout << "DEBUG:  tauCheck:   " << tauCheck_ << "\n";
  cout << "DEBUG:  jetCheck:   " << jetCheck_ << "\n";
  cout << "DEBUG:  sumCheck:   " << sumCheck_ << "\n";
  cout << "DEBUG:  muonCheck:  " << muonCheck_ << "\n";

  if (egCheck_)   {egToken_   = consumes<EGammaBxCollection> (iConfig.getParameter<InputTag>("egToken"));}
  if (tauCheck_)  {tauToken_  = consumes<TauBxCollection>    (iConfig.getParameter<InputTag>("tauToken"));}
  if (jetCheck_)  {jetToken_  = consumes<JetBxCollection>    (iConfig.getParameter<InputTag>("jetToken"));}
  if (sumCheck_)  {sumToken_  = consumes<EtSumBxCollection>  (iConfig.getParameter<InputTag>("sumToken"));}
  if (muonCheck_) {muonToken_ = consumes<MuonBxCollection>   (iConfig.getParameter<InputTag>("muonToken"));}

}

L1TSummary::~L1TSummary(){}


void 
L1TSummary::analyze(Event const& iEvent, EventSetup const& iSetup)
{
  cout << "L1TSummary Module output for " << tag_ << "\n";
  if (egCheck_){
    Handle<EGammaBxCollection> XTMP;    
    iEvent.getByToken(egToken_, XTMP);
    if (XTMP.isValid()){ 
      cout << "INFO:  L1T found e-gamma collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
	for (auto it=XTMP->begin(ibx); it!=XTMP->end(ibx); it++){      
	  if (it->et() > 0) 
	    cout << "bx:  " << ibx << "  et:  "  << it->et() << "  eta:  "  << it->eta() << "  phi:  "  << it->phi() << "\n";
	}
      }
    } else {
      LogWarning("MissingProduct") << "L1Upgrade e-gamma's not found." << std::endl;
    }
  }

  if (tauCheck_){
    Handle<TauBxCollection> XTMP;    
    iEvent.getByToken(tauToken_, XTMP);
    if (XTMP.isValid()){ 
      cout << "INFO:  L1T found tau collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
	for (auto it=XTMP->begin(ibx); it!=XTMP->end(ibx); it++){      
	  if (it->et() > 0) 
	    cout << "bx:  " << ibx << "  et:  "  << it->et() << "  eta:  "  << it->eta() << "  phi:  "  << it->phi() << "\n";
	}
      }
    } else {
      LogWarning("MissingProduct") << "L1Upgrade tau's not found." << std::endl;
    }
  }

  if (jetCheck_){
    Handle<JetBxCollection> XTMP;    
    iEvent.getByToken(jetToken_, XTMP);
    if (XTMP.isValid()){ 
      cout << "INFO:  L1T found jet collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
	for (auto it=XTMP->begin(ibx); it!=XTMP->end(ibx); it++){      
	  if (it->et() > 0) 
	    cout << "bx:  " << ibx << "  et:  "  << it->et() << "  eta:  "  << it->eta() << "  phi:  "  << it->phi() << "\n";
	}
      }
    } else {
      LogWarning("MissingProduct") << "L1T upgrade jets not found." << std::endl;
    }
  }

  if (sumCheck_){
    Handle<EtSumBxCollection> XTMP;    
    iEvent.getByToken(sumToken_, XTMP);
    if (XTMP.isValid()){ 
      cout << "INFO:  L1T found sum collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
	for (auto it=XTMP->begin(ibx); it!=XTMP->end(ibx); it++){      
	  if (it->et() > 0) 
	    cout << "bx:  " << ibx << "  et:  "  << it->et() << "  eta:  "  << it->eta() << "  phi:  "  << it->phi() << "\n";
	}
      }
    } else {
      LogWarning("MissingProduct") << "L1T upgrade sums not found." << std::endl;
    }
  }
 

  if (muonCheck_){
    Handle<MuonBxCollection> XTMP; 
    iEvent.getByToken(muonToken_, XTMP);
    if (XTMP.isValid()){ 
      cout << "INFO:  L1T found muon collection.\n";
      for (int ibx = XTMP->getFirstBX(); ibx <= XTMP->getLastBX(); ++ibx) {
	for (auto it=XTMP->begin(ibx); it!=XTMP->end(ibx); it++){      
	  if (it->et() > 0) 
	    cout << "bx:  " << ibx << "  et:  "  << it->et() << "  eta:  "  << it->eta() << "  phi:  "  << it->phi() << "\n";
	}
      }
    } else {
      LogWarning("MissingProduct") << "L1T upgrade muons not found." << std::endl;
    }
  }





}

void
L1TSummary::beginJob()
{
  cout << "INFO:  L1TSummary module beginJob called.\n";
}

void
L1TSummary::endJob() {
  cout << "INFO:  L1TSummary module endJob called.\n";
}

void
L1TSummary::beginRun(Run const& run, EventSetup const& iSetup)
{
}

void
L1TSummary::endRun(Run const&, EventSetup const&)
{
}

void
L1TSummary::beginLuminosityBlock(LuminosityBlock const&, EventSetup const&)
{
}

void
L1TSummary::endLuminosityBlock(LuminosityBlock const&, EventSetup const&)
{
}

void
L1TSummary::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


DEFINE_FWK_MODULE(L1TSummary);
