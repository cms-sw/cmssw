// -*- C++ -*-
//
// Package:     TopMuonObject
// Class  :     TopMuonObject
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TopMuonObject.cc,v 1.1 2007/05/02 15:05:04 lowette Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopMuonObject.h"

TopMuonObject::TopMuonObject(){ }
TopMuonObject::TopMuonObject(TopMuon rm){ recMuon = rm; }
TopMuonObject::~TopMuonObject(){ }

void 		TopMuonObject::setRecMuon(TopMuon rm)       	{ recMuon = rm; }
void 		TopMuonObject::setFitMuon(TopParticle fm) 	{ fitMuon = fm; }      
//void    	TopMuonObject::setLRvalue(double lv)		{ LRvalue = lv; }


TopMuon		TopMuonObject::getRecMuon() const	  	{ return recMuon; }
TopParticle	TopMuonObject::getFitMuon() const	  	{ return fitMuon; }    
//double    	TopMuonObject::getLRvalue() const		{ return LRvalue; }
