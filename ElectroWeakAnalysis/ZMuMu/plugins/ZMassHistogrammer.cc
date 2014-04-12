#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "TH1.h"

class ZMassHistogrammer : public edm::EDAnalyzer {
public:
  ZMassHistogrammer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  edm::EDGetTokenT<reco::CandidateView>  zToken_;
  edm::EDGetTokenT<reco::CandidateView>  genToken_;
  TH1F *h_mZ_, *h_mZMC_;
};

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace edm;

ZMassHistogrammer::ZMassHistogrammer(const ParameterSet& pset) :
  zToken_(consumes<reco::CandidateView>(pset.getParameter<InputTag>("z"))),
  genToken_(consumes<reco::CandidateView>(pset.getParameter<InputTag>("gen"))) {
  cout << ">>> Z Mass constructor" << endl;
  Service<TFileService> fs;
  h_mZ_ = fs->make<TH1F>("ZMass", "Z mass (GeV/c^{2})", 100,  0, 200);
  h_mZMC_ = fs->make<TH1F>("ZMCMass", "Z MC mass (GeV/c^{2})", 100,  0, 200);
}

void ZMassHistogrammer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  cout << ">>> Z Mass analyze" << endl;
  Handle<CandidateCollection> z;
  Handle<CandidateCollection> gen;
  event.getByToken(zToken_, z);
  event.getByToken(genToken_, gen);
  for(unsigned int i = 0; i < z->size(); ++i) {
    const Candidate &zCand = (*z)[i];
    h_mZ_->Fill(zCand.mass());
  }
  for(unsigned int i = 0; i < gen->size(); ++i) {
    const Candidate &genCand = (*gen)[i];
    if((genCand.pdgId() == 23) && (genCand.status() == 2)) //this is an intermediate Z0
      cout << ">>> intermediate Z0 found, with " << genCand.numberOfDaughters()
	   << " daughters" << endl;
    if((genCand.pdgId() == 23)&&(genCand.status() == 3)) { //this is a Z0
      cout << ">>> Z0 found, with " << genCand.numberOfDaughters()
	   << " daughters" << endl;
      h_mZMC_->Fill(genCand.mass());
      if(genCand.numberOfDaughters() == 3) {//Z0 decays in mu+ mu-, the 3rd daughter is the same Z0
	const Candidate * dauGen0 = genCand.daughter(0);
	const Candidate * dauGen1 = genCand.daughter(1);
	const Candidate * dauGen2 = genCand.daughter(2);
	cout << ">>> daughter MC 0 PDG Id " << dauGen0->pdgId()
	     << ", status " << dauGen0->status()
	     << ", charge " << dauGen0->charge()
	     << endl;
	cout << ">>> daughter MC 1 PDG Id " << dauGen1->pdgId()
	     << ", status " << dauGen1->status()
	     << ", charge " << dauGen1->charge()
	     << endl;
	cout << ">>> daughter MC 2 PDG Id " << dauGen2->pdgId()
	     << ", status " << dauGen2->status()
	     << ", charge " << dauGen2->charge() << endl;
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMassHistogrammer);

