#include "JetMETCorrections/Algorithms/interface/L2RelativeCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL2RelativeCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace std;


L2RelativeCorrector::L2RelativeCorrector (const edm::ParameterSet& fConfig) {
 std::string file="CondFormats/JetMETObjects/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleL2RelativeCorrector (f1.fullPath());
}

L2RelativeCorrector::~L2RelativeCorrector () {
  delete mSimpleCorrector;
} 

double L2RelativeCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionPtEta(fJet.pt(), fJet.eta());
}

double L2RelativeCorrector::correction (const reco::Jet& fJet) const {
  return correction (fJet.p4());
}
