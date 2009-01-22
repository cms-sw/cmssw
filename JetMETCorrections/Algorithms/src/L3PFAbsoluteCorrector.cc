#include "JetMETCorrections/Algorithms/interface/L3PFAbsoluteCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL3PFAbsoluteCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace std;


L3PFAbsoluteCorrector::L3PFAbsoluteCorrector (const edm::ParameterSet& fConfig) {
 std::string file="CondFormats/JetMETObjects/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleL3PFAbsoluteCorrector (f1.fullPath());
}

L3PFAbsoluteCorrector::~L3PFAbsoluteCorrector () {
  delete mSimpleCorrector;
} 

double L3PFAbsoluteCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionPtEta(fJet.pt(), fJet.eta());
}

double L3PFAbsoluteCorrector::correction (const reco::Jet& fJet) const {
  return correction (fJet.p4());
}
