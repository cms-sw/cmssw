// -*- C++ -*-
//
// Package:     TopMETObject
// Class  :     TopMETObject
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TopMETObject.cc,v 1.2 2007/05/01 14:44:00 heyninck Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopMETObject.h"

TopMETObject::TopMETObject(){ }
TopMETObject::TopMETObject(TopMET rm){ recMET = rm; }
TopMETObject::~TopMETObject(){ }

void 		TopMETObject::setRecMET(TopMET rm)        { recMET = rm; }
void 		TopMETObject::setFitMET(TopParticle fm)   { fitMET = fm; }

TopMET		TopMETObject::getRecMET() const	  	  { return recMET; }
TopParticle	TopMETObject::getFitMET() const	  	  { return fitMET; }
