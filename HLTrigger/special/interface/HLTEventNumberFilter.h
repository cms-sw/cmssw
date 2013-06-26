#ifndef HLTEventNumberFilter_h
#define HLTEventNumberFilter_h
// -*- C++ -*-
//
// Package:    HLTEventNumberFilter
// Class:      HLTEventNumberFilter
// 
/**\class HLTEventNumberFilter HLTEventNumberFilter.cc filter/HLTEventNumberFilter/src/HLTEventNumberFilter.cc

Description: Filter to select HCAL abort gap events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Martin Grunewald
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTEventNumberFilter.h,v 1.2 2012/01/21 15:00:13 fwyzard Exp $
//
//


// include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include <string>

//
// class declaration
//

class HLTEventNumberFilter : public HLTFilter {
public:
  explicit HLTEventNumberFilter(const edm::ParameterSet&);
  virtual ~HLTEventNumberFilter();
  
private:
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
  
  // ----------member data ---------------------------

  /// accept the event if its event number is a multiple of period_
  unsigned int period_;
  /// if invert_=true, invert that event accept decision
  bool invert_;

};

#endif
