// -*- C++ -*-
//
// Package:     StGenEvent
// Class  :     StGenEvent
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr 10 11:48:25 CEST 2007
// $Id: StGenEvent.cc,v 1.1 2007/05/02 15:05:04 lowette Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"

StGenEvent::StGenEvent(){}

StGenEvent::StGenEvent(int dec,vector<const Candidate*> ps){
  decay_ = dec;
  for(unsigned int i= 0; i<ps.size(); i++) particles_.push_back( (Candidate*) ps[i]);
}

StGenEvent::~StGenEvent() {}
