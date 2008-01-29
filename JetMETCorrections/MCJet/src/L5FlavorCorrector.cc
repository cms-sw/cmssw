//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: L5FlavorCorrector.cc,v 1.5 2007/03/30 23:47:55 fedor Exp $
//
// L5 Jet flavor Corrector
//
#include "JetMETCorrections/MCJet/interface/L5FlavorCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL5FlavorCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace std;


L5FlavorCorrector::L5FlavorCorrector (const edm::ParameterSet& fConfig) {
 std::string file="JetMETCorrections/MCJet/data/"+fConfig.getParameter <std::string> ("tagName")+".txt";
 edm::FileInPath f1(file);
 mSimpleCorrector = new SimpleL5FlavorCorrector (f1.fullPath());
}

L5FlavorCorrector::~L5FlavorCorrector () {
  delete mSimpleCorrector;
} 

double L5FlavorCorrector::correction (const LorentzVector& fJet) const {
  return mSimpleCorrector->correctionPtEta (fJet.Pt(), fJet.Eta());
}
