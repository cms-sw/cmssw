#ifndef HLTTriggerTypeFilter_h
#define HLTTriggerTypeFilter_h
// -*- C++ -*-
//
// Package:    HLTTriggerTypeFilter
// Class:      HLTTriggerTypeFilter
// 
/**\class HLTTriggerTypeFilter HLTTriggerTypeFilter.cc 

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTTriggerTypeFilter.h,v 1.4 2012/01/22 22:20:49 fwyzard Exp $
//
//


// include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class declaration
//

class HLTTriggerTypeFilter : public edm::EDFilter {
public:
  explicit HLTTriggerTypeFilter(const edm::ParameterSet&);
  ~HLTTriggerTypeFilter();
  
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
  // ----------member data ---------------------------  
  unsigned short  SelectedTriggerType_;
  
};

#endif
