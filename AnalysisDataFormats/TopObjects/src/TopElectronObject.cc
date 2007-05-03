// -*- C++ -*-
//
// Package:     TopElectronObject
// Class  :     TopElectronObject
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TopElectronObject.cc,v 1.1 2007/05/02 15:05:04 lowette Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopElectronObject.h"

TopElectronObject::TopElectronObject(){ }
TopElectronObject::TopElectronObject(TopElectron ce){ recElectron = ce; }
TopElectronObject::~TopElectronObject(){ }

void 		TopElectronObject::setRecElectron(TopElectron re)      	{ recElectron = re; }
void 		TopElectronObject::setFitElectron(TopParticle fe) 	{ fitElectron = fe; }     
//void    	TopElectronObject::setLRvalue(double lv)			{ LRvalue = lv; }

TopElectron	TopElectronObject::getRecElectron() const	  	{ return recElectron; }
TopParticle	TopElectronObject::getFitElectron() const	  	{ return fitElectron; }   
//double    	TopElectronObject::getLRvalue() const			{ return LRvalue; }
