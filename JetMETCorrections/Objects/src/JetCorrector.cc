//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: JetCorrector.cc,v 1.6 2009/11/10 14:47:33 schiefer Exp $
//
// Generic interface for JetCorrection services
//

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

double JetCorrector::correction (const reco::Jet& fJet, 
				 const edm::RefToBase<reco::Jet>& fJetRef,
				 const edm::Event& fEvent,
				 const edm::EventSetup& fSetup) const {
  if (eventRequired ()) {
    edm::LogError ("Missing Jet Correction Method") 
      << "Undefined Jet Correction method requiring event data is called" << std::endl;
    return 0;
  }
  return correction (fJet);
}

double JetCorrector::correction (const reco::Jet& fJet, 
				 const edm::RefToBase<reco::Jet>& fJetRef,
				 const edm::Event& fEvent, 
				 const edm::EventSetup& fSetup,
				 LorentzVector& corrected ) const {
  if ( vectorialCorrection() ) {
    edm::LogError ("Missing Jet Correction Method") 
      << "Undefined Jet (vectorial) correction method requiring event data is called" << std::endl;
    return 0;
  }
  return correction (fJet);
}

const JetCorrector* JetCorrector::getJetCorrector (const std::string& fName, const edm::EventSetup& fSetup) {
  const JetCorrectionsRecord& record = fSetup.get <JetCorrectionsRecord> ();
  edm::ESHandle <JetCorrector> handle;
  record.get (fName, handle);
  return &*handle;
}
