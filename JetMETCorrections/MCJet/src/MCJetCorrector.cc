//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: MCJetCorrector.cc,v 1.4 2007/02/26 20:31:26 fedor Exp $
//
// MC Jet Corrector
//
#include "JetMETCorrections/MCJet/interface/MCJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleMCJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"

using namespace std;


MCJetCorrector::MCJetCorrector (const edm::ParameterSet& fConfig) {
 std::string file="JetMETCorrections/MCJet/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleMCJetCorrector (f1.fullPath());
}

MCJetCorrector::~MCJetCorrector () {
  delete mSimpleCorrector;
} 

double MCJetCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionPtEtaPhiE (fJet.Pt(), fJet.Eta(), fJet.Phi(),fJet.E());
}
