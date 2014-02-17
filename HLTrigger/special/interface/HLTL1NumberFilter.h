#ifndef HLTL1NumberFilter_h
#define HLTL1NumberFilter_h
// -*- C++ -*-
//
// Package:    HLTL1NumberFilter
// Class:      HLTL1NumberFilter
// 
/**\class HLTL1NumberFilter HLTL1NumberFilter.cc filter/HLTL1NumberFilter/src/HLTL1NumberFilter.cc

Description: Filter to select HCAL abort gap events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Martin Grunewald
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTL1NumberFilter.h,v 1.3 2012/01/23 00:00:15 fwyzard Exp $
//
//


// include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

//
// class declaration
//

class HLTL1NumberFilter : public edm::EDFilter {
public:
  explicit HLTL1NumberFilter(const edm::ParameterSet&);
  virtual ~HLTL1NumberFilter();
  
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
  // ----------member data ---------------------------

  /// raw data
  edm::InputTag input_ ; 
  /// accept the event if its event number is a multiple of period_
  unsigned int period_;
  /// if invert_=true, invert that event accept decision
  bool invert_;

};

#endif
