// -*- C++ -*-
//
// Package:     TtGenEvent
// Class  :     TtGenEvent
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr 10 11:48:25 CEST 2007
// $Id: TtGenEvent.cc,v 1.5 2007/04/30 10:43:30 heyninck Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

TtGenEvent::TtGenEvent(){}

TtGenEvent::TtGenEvent(int dec,vector<const Candidate*> ps){
  decay_ = dec;
  for(unsigned int i= 0; i<ps.size(); i++) particles_.push_back( (Candidate*) ps[i]);
}

TtGenEvent::~TtGenEvent() {}
