//
// Original Author:  Olga Kodolova, September 2007
//
// ZSP Jet Corrector
//
#include "JetMETCorrections/Algorithms/interface/ZSPJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleZSPJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"

using namespace std;


ZSPJetCorrector::ZSPJetCorrector (const edm::ParameterSet& fConfig) {
 std::string file="JetMETCorrections/ZSPJet/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleZSPJetCorrector (f1.fullPath());
}

ZSPJetCorrector::~ZSPJetCorrector () {
  delete mSimpleCorrector;
} 

double ZSPJetCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionPtEtaPhiE (fJet.Pt(), fJet.Eta(), fJet.Phi(),fJet.E());
}
