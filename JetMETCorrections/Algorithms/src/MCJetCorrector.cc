//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: MCJetCorrector.cc,v 1.1 2007/10/03 23:29:51 fedor Exp $
//
// MC Jet Corrector
//
#include "JetMETCorrections/Algorithms/interface/MCJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleMCJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"

using namespace std;


MCJetCorrector::MCJetCorrector (const edm::ParameterSet& fConfig) {
 std::string file=fConfig.getParameter <std::string> ("tagName");
 if (file.find (".txt") == std::string::npos) { // reference by type
   file = "CondFormats/JetMETObjects/data/" + file + ".txt";
 }
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleMCJetCorrector (f1.fullPath());
}

MCJetCorrector::~MCJetCorrector () {
  delete mSimpleCorrector;
} 

double MCJetCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionPtEtaPhiE (fJet.Pt(), fJet.Eta(), fJet.Phi(),fJet.E());
}

double MCJetCorrector::correction (const reco::Jet& fJet) const {
  return correction (fJet.p4 ());
}

