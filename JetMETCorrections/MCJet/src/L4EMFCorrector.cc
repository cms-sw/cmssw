//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: L4EMFCorrector.cc,v 1.1 2007/11/14 00:03:32 fedor Exp $
//
// L4 EMF Corrector
//
#include "JetMETCorrections/MCJet/interface/L4EMFCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL4EMFCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace std;


L4EMFCorrector::L4EMFCorrector (const edm::ParameterSet& fConfig) {
 std::string file="CondFormats/JetMETObjects/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleL4EMFCorrector (f1.fullPath());
}

L4EMFCorrector::~L4EMFCorrector () {
  delete mSimpleCorrector;
} 

double L4EMFCorrector::correction (const LorentzVector& fJet) const {
  throw cms::Exception("Invalid jet type") << "L4EMFCorrector is applicable to CaloJets only";
  return 1;
}

double L4EMFCorrector::correction (const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const {
  const reco::CaloJet& caloJet = dynamic_cast <const reco::CaloJet&> (fJet); // applicable to CaloJets only
  return mSimpleCorrector->correctionPtEtaEmfraction (caloJet.pt(), caloJet.eta(), caloJet.emEnergyFraction());
}
