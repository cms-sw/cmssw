// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <TMath.h>
#include <TFile.h>
#include <TH1.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

using namespace std;
using namespace reco;
using namespace edm;

class printPartonJet : public edm::one::EDAnalyzer<> {
public:
  explicit printPartonJet(const edm::ParameterSet&);
  ~printPartonJet(){};
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  edm::EDGetTokenT<View<Candidate> > sourceToken_;
  string fOutputFileName_;
  Handle<View<Candidate> > partonJets;
};

printPartonJet::printPartonJet(const edm::ParameterSet& iConfig) {
  sourceToken_ = consumes<View<Candidate> >(iConfig.getParameter<InputTag>("src"));
}

void printPartonJet::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cout << "[printPartonJet] analysing event " << iEvent.id() << endl;

  try {
    iEvent.getByToken(sourceToken_, partonJets);
  } catch (std::exception& ce) {
    cerr << "[printPartonJet] caught std::exception " << ce.what() << endl;
    return;
  }

  cout << "************************" << endl;
  cout << "* PartonJetCollection  *" << endl;
  cout << "************************" << endl;
  for (size_t j = 0; j != partonJets->size(); ++j) {
    printf("[printPartonJet] (pt,eta,phi) = %7.3f %6.3f %6.3f |\n",
           (*partonJets)[j].et(),
           (*partonJets)[j].eta(),
           (*partonJets)[j].phi());
    for (Candidate::const_iterator itC = (*partonJets)[j].begin(); itC != (*partonJets)[j].end(); itC++) {
      cout << "              Constituent (pt,eta,phi,pdgId): " << itC->pt() << " " << itC->eta() << " " << itC->phi()
           << " " << itC->pdgId() << endl;
    }
  }
}

DEFINE_FWK_MODULE(printPartonJet);
