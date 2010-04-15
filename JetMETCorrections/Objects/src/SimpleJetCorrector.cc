//
// Original Author:  Fedor Ratnikov Feb. 16, 2007
// $Id: JetCorrector.h,v 1.3 2007/01/18 01:35:13 fedor Exp $
//
// Simplest jet corrector scaling every jet by fixed factor
//

#include "JetMETCorrections/Objects/interface/SimpleJetCorrector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

SimpleJetCorrector::SimpleJetCorrector (const edm::ParameterSet& fConfig) 
  : mScale (fConfig.getParameter <double> ("scale"))
{}

SimpleJetCorrector::~SimpleJetCorrector () {}
