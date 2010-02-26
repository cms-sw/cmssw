//
// Original Author:  Attilio Santocchia Feb. 28, 2008
//
// L7 Jet Parton Corrector
//
#include "JetMETCorrections/Algorithms/interface/L7PartonCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL7PartonCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace std;


L7PartonCorrector::L7PartonCorrector (const edm::ParameterSet& fConfig) {
 std::string file="CondFormats/JetMETObjects/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 std::string section=fConfig.getParameter <std::string> ("section");

 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleL7PartonCorrector (f1.fullPath(), section);
}

L7PartonCorrector::~L7PartonCorrector () {
  delete mSimpleCorrector;
} 

double L7PartonCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionPtEta (fJet.Pt(), fJet.Eta());
}

double L7PartonCorrector::correction (const reco::Jet& fJet) const {
  return correction (fJet.p4());
}
