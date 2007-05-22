// -*- C++ -*-
//
// Package:     TopJet
// Class  :     TopJet
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TopJet.cc,v 1.2 2007/05/04 01:08:38 lowette Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"

TopJet::TopJet(){ }
TopJet::TopJet(JetType aJet): TopObject<JetType>(aJet) {}
TopJet::~TopJet(){ }

void 		TopJet::setGenJet(reco::Particle gj)  	{ genJet = gj; }
void 		TopJet::setRecJet(JetType rj)     	{ recJet = rj; }
void 		TopJet::setFitJet(TopParticle fj) 	{ fitJet = fj; }
void 		TopJet::setBdiscriminant(double b)	{ bdiscr = b; }

reco::Particle	TopJet::getGenJet() const	  	{ return genJet; }
JetType		TopJet::getRecJet() const	  	{ return recJet; }
TopParticle	TopJet::getFitJet() const	  	{ return fitJet; }
double   	TopJet::getBdiscriminant() const 	{ return bdiscr; }
