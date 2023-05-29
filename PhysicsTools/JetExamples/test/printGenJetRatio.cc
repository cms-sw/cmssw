// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/JetMatching/interface/JetFlavour.h"
#include "DataFormats/JetMatching/interface/JetFlavourMatching.h"

class printGenJetRatio : public edm::one::EDAnalyzer<> {
public:
  typedef reco::JetFloatAssociation::Container JetBCEnergyRatioCollection;

  explicit printGenJetRatio(const edm::ParameterSet&);
  ~printGenJetRatio(){};
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  edm::EDGetTokenT<JetBCEnergyRatioCollection> sourceBratioToken_;
  edm::EDGetTokenT<JetBCEnergyRatioCollection> sourceCratioToken_;
  edm::Handle<JetBCEnergyRatioCollection> theBratioValue;
  edm::Handle<JetBCEnergyRatioCollection> theCratioValue;
};

// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace reco;
using namespace edm;

printGenJetRatio::printGenJetRatio(const edm::ParameterSet& iConfig) {
  sourceBratioToken_ = consumes<JetBCEnergyRatioCollection>(iConfig.getParameter<InputTag>("srcBratio"));
  sourceCratioToken_ = consumes<JetBCEnergyRatioCollection>(iConfig.getParameter<InputTag>("srcCratio"));
}

void printGenJetRatio::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cout << "[printGenJetRatio] analysing event " << iEvent.id() << endl;

  try {
    iEvent.getByToken(sourceBratioToken_, theBratioValue);
    iEvent.getByToken(sourceCratioToken_, theCratioValue);
  } catch (std::exception& ce) {
    cerr << "[printJetFlavour] caught std::exception " << ce.what() << endl;
    return;
  }

  cout << "-------------------- GenJet Bratio ------------------------" << endl;
  for (JetBCEnergyRatioCollection::const_iterator itB = theBratioValue->begin(); itB != theBratioValue->end(); itB++) {
    const Jet& jetB = *(itB->first);
    float cR = 0;
    for (JetBCEnergyRatioCollection::const_iterator itC = theCratioValue->begin(); itC != theCratioValue->end();
         itC++) {
      if (itB->first == itC->first)
        cR = itC->second;
    }
    printf("printGenJetRatio] (pt,eta,phi) jet = %7.3f %6.3f %6.3f | bcRatio = %7.5f - %7.5f \n",
           jetB.et(),
           jetB.eta(),
           jetB.phi(),
           itB->second,
           cR);
  }
}

DEFINE_FWK_MODULE(printGenJetRatio);
