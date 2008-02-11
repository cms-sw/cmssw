//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: PUJetCorrector.cc,v 1.2 2007/11/01 21:52:54 fedor Exp $
//
// PU Jet Corrector
//
#include "JetMETCorrections/Algorithms/interface/PUJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimplePUJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"

using namespace std;


PUJetCorrector::PUJetCorrector (const edm::ParameterSet& fConfig) {
 std::string file=fConfig.getParameter <std::string> ("tagName");
 if (file.find (".txt") == std::string::npos) { // reference by type
   file = "CondFormats/JetMETObjects/data/" + file + ".txt";
 }
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimplePUJetCorrector (f1.fullPath());
}

PUJetCorrector::~PUJetCorrector () {
  delete mSimpleCorrector;
} 

double PUJetCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionPtEtaPhiE (fJet.Pt(), fJet.Eta(), fJet.Phi(),fJet.E());
}

double PUJetCorrector::correction (const reco::Jet& fJet) const {
  return correction (fJet.p4 ());
}

