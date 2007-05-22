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
// $Id: TtGenEvent.cc,v 1.1 2007/05/02 15:05:04 lowette Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

TtGenEvent::TtGenEvent(){}

TtGenEvent::TtGenEvent(int dec,std::vector<const reco::Candidate*> ps){
  decay_ = dec;
  for(unsigned int i= 0; i<ps.size(); i++) particles_.push_back( (reco::Candidate*) ps[i]);
}

TtGenEvent::~TtGenEvent() {}
