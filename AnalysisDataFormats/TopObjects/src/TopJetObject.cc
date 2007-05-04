// -*- C++ -*-
//
// Package:     TopJetObject
// Class  :     TopJetObject
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TopJetObject.cc,v 1.1 2007/05/02 15:05:04 lowette Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopJetObject.h"

TopJetObject::TopJetObject(){ }
TopJetObject::TopJetObject(JetType rj){ recJet = rj; }
TopJetObject::~TopJetObject(){ }

void 		TopJetObject::setRecJet(JetType rj)     { recJet = rj; }
void 		TopJetObject::setLCalJet(TopJet lcj)    { lCalJet = lcj; }
void 		TopJetObject::setBCalJet(TopJet bcj)    { bCalJet = bcj; }
void 		TopJetObject::setFitJet(TopParticle fj) { fitJet = fj; }
void 		TopJetObject::setBdiscriminant(double b){ bdiscr = b; }

JetType		TopJetObject::getRecJet() const	  	{ return recJet; }
TopJet		TopJetObject::getLCalJet() const	{ return lCalJet; }
TopJet		TopJetObject::getBCalJet() const	{ return bCalJet; }
TopParticle	TopJetObject::getFitJet() const	  	{ return fitJet; }
double   	TopJetObject::getBdiscriminant() const 	{ return bdiscr; }
