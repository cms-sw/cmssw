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
// $Id: StGenEvent.cc,v 1.1 2007/05/11 15:40:47 giamman Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"

StGenEvent::StGenEvent(){}

StGenEvent::StGenEvent(int dec,std::vector<const reco::Candidate*> ps){
  decay_ = dec;
  for(unsigned int i= 0; i<ps.size(); i++) particles_.push_back( (reco::Candidate*) ps[i]);
}

StGenEvent::~StGenEvent() {}
