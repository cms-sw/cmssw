// Generic interface for METCorrection services
//

#include "JetMETCorrections/Objects/interface/METCorrector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "JetMETCorrections/Objects/interface/METCorrectionsRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


double METCorrector::correction (const reco::Jet& fJet,
				 const edm::Event& fEvent,
				 const edm::EventSetup& fSetup) const {
  // Added by Sasha (begin)
  //std::cout <<  " JetCorrector::correction(1) " << std::endl;
  // Added by Sasha (end)
  if (eventRequired () && !refRequired()) {
    edm::LogError ("Missing Jet Correction Method") 
      << "Undefined Jet Correction method requiring event data is called" << std::endl;
    // Added by Sasha (begin)
    //std::cout <<  " JetCorrector::Returned 0 " << std::endl;
    // Added by Sasha (end)
    return 0;
  }
  // Added by Sasha (begin)
  //std::cout << "JetCorrector Correction1  " << correction (fJet) << std::endl;
  // Added by Sasha (end)
  return correction (fJet);
}


double METCorrector::correction (const reco::Jet& fJet, 
				 const edm::RefToBase<reco::Jet>& fJetRef,
				 const edm::Event& fEvent,
				 const edm::EventSetup& fSetup) const {
  //std::cout <<  " JetCorrector::2 " << std::endl;
  if (eventRequired () && refRequired()) {
    edm::LogError ("Missing Jet Correction Method") 
      << "Undefined Jet Correction method requiring event data and jet reference is called" << std::endl;
    return 0;
  }
  return correction (fJet);
}

double METCorrector::correction (const reco::Jet& fJet, 
				 const edm::RefToBase<reco::Jet>& fJetRef,
				 const edm::Event& fEvent, 
				 const edm::EventSetup& fSetup,
				 LorentzVector& corrected ) const {
  std::cout <<  " JetCorrector::3 " << std::endl;
  if ( vectorialCorrection() ) {
    edm::LogError ("Missing Jet Correction Method") 
      << "Undefined Jet (vectorial) correction method requiring event data is called" << std::endl;
    return 0;
  }
  return correction (fJet);
}

double METCorrector::correction (const reco::MET& fMet,
				 const edm::Event& fEvent,
				 const edm::EventSetup& fSetup) const {
  // Added by Sasha (begin)
  //std::cout <<  " JetCorrector::correction(1) " << std::endl;
  // Added by Sasha (end)
  if (eventRequired () && !refRequired()) {
    edm::LogError ("Missing Jet Correction Method") 
      << "Undefined Jet Correction method requiring event data is called" << std::endl;
    // Added by Sasha (begin)
    //std::cout <<  " JetCorrector::Returned 0 " << std::endl;
    // Added by Sasha (end)
    return 0;
  }
  // Added by Sasha (begin)
  //std::cout << "JetCorrector Correction1  " << correction (fJet) << std::endl;
  // Added by Sasha (end)
  //return correction (fMet);
  return 90;
}

/*
const JetCorrector* METCorrector::getJetCorrector (const std::string& fName, const edm::EventSetup& fSetup) {
  const JetCorrectionsRecord& record = fSetup.get <JetCorrectionsRecord> ();
  edm::ESHandle <JetCorrector> handle;
  record.get (fName, handle);
  return &*handle;
}
*/
