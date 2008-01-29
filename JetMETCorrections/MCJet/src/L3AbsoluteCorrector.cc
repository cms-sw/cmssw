#include "JetMETCorrections/MCJet/interface/L3AbsoluteCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL3AbsoluteCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace std;


L3AbsoluteCorrector::L3AbsoluteCorrector (const edm::ParameterSet& fConfig) {
 std::string file="JetMETCorrections/MCJet/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleL3AbsoluteCorrector (f1.fullPath());
}

L3AbsoluteCorrector::~L3AbsoluteCorrector () {
  delete mSimpleCorrector;
} 

double L3AbsoluteCorrector::correction (const LorentzVector& fJet) const {
  throw cms::Exception("Invalid jet type") << "L3AbsoluteCorrector is applicable to CaloJets only";
  return 1;
}

double L3AbsoluteCorrector::correction (const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const {
  const reco::CaloJet& caloJet = dynamic_cast <const reco::CaloJet&> (fJet); // applicable to CaloJets only
  return mSimpleCorrector->correctionPtEta(caloJet.pt(), caloJet.eta());
}
