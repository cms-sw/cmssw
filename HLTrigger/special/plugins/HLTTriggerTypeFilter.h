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
//
//


// include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// class declaration
//

class HLTTriggerTypeFilter : public edm::global::EDFilter<> {
public:
  explicit HLTTriggerTypeFilter(const edm::ParameterSet&);
  ~HLTTriggerTypeFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:
  bool filter(edm::StreamID, edm::Event &, edm::EventSetup const &) const final;
  
  // ----------member data ---------------------------  
  unsigned short  selectedTriggerType_;
  
};

#endif
