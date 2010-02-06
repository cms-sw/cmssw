#include "JetMETCorrections/Algorithms/interface/L1OffsetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL1OffsetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace std;


L1OffsetCorrector::L1OffsetCorrector (const edm::ParameterSet& fConfig) {
 std::string file="CondFormats/JetMETObjects/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleL1OffsetCorrector (f1.fullPath());
}

L1OffsetCorrector::~L1OffsetCorrector () {
  delete mSimpleCorrector;
} 

double L1OffsetCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionEnEta(fJet.energy(), fJet.eta());
}

double L1OffsetCorrector::correction (const reco::Jet& fJet) const {
  return correction (fJet.p4());
}
