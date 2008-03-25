//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: MCJetCorrector3D.cc,v 1.1 2007/11/01 21:52:54 fedor Exp $
//
// MC Jet Corrector
//
#include "JetMETCorrections/MCJet/interface/MCJetCorrector3D.h"
#include "CondFormats/JetMETObjects/interface/Simple3DMCJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"


MCJetCorrector3D::MCJetCorrector3D (const edm::ParameterSet& fConfig) {
 std::string file=fConfig.getParameter <std::string> ("tagName");
 if (file.find (".txt") == std::string::npos) { // reference by type
   file = "JetMETCorrections/MCJet/data/" + file + ".txt";
 }
 edm::FileInPath f1(file);
 mSimpleCorrector = new Simple3DMCJetCorrector (f1.fullPath());
}

MCJetCorrector3D::~MCJetCorrector3D () {
  delete mSimpleCorrector;
} 

double MCJetCorrector3D::correction (const LorentzVector& fJet) const {
  throw cms::Exception("Invalid jet type") << "MCJetCorrector3D is applicable to CaloJets only";
  return 1;
}

double MCJetCorrector3D::correction (const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const {
  const reco::CaloJet& caloJet = dynamic_cast <const reco::CaloJet&> (fJet); // applicable to CaloJets only
  return mSimpleCorrector->correctionPtEtaEmfraction (caloJet.pt(), caloJet.eta(), caloJet.emEnergyFraction());
}
