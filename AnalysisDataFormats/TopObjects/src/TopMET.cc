// -*- C++ -*-
//
// Package:     TopMET
// Class  :     TopMET
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TopMET.cc,v 1.1 2007/05/02 15:05:04 lowette Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"

TopMET::TopMET(){ }
TopMET::TopMET(METType aMet): TopObject<METType>(aMet) {}
TopMET::~TopMET(){ }

void 		TopMET::setGenMET(reco::Particle gm)     { genMET = gm; }
void 		TopMET::setFitMET(TopParticle fm)   { fitMET = fm; }

reco::Particle	TopMET::getGenMET() const	    { return genMET; }
TopParticle	TopMET::getFitMET() const	    { return fitMET; }
