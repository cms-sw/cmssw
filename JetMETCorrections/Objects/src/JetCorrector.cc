//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: HcalHardcodeCalibrations.h,v 1.5 2006/01/10 19:29:40 fedor Exp $
//
// Generic interface for JetCorrection services
//

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

double JetCorrector::correction (const reco::Jet& fJet) const {
  edm::LogError ("Missing Jet Correction Method") << "Undefined eventless Jet Correction method is called" << std::endl; 
  return 0;
}

double JetCorrector::correction (const reco::Jet& fJet, 
				 const edm::Event& fEvent, 
				 const edm::EventSetup& fSetup) const {
  if (eventRequired ()) {
    edm::LogError ("Missing Jet Correction Method") << "Undefined Jet Correction method requiring event data is called" << std::endl;
    return 0;
  }
  return correction (fJet);
}
