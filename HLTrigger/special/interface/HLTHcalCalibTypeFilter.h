#ifndef HLTHcalCalibTypeFilter_h
#define HLTHcalCalibTypeFilter_h
// -*- C++ -*-
//
// Package:    HLTHcalCalibTypeFilter
// Class:      HLTHcalCalibTypeFilter
// 
/**\class HLTHcalCalibTypeFilter HLTHcalCalibTypeFilter.cc filter/HLTHcalCalibTypeFilter/src/HLTHcalCalibTypeFilter.cc

Description: Filter to select HCAL abort gap events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Bryan DAHMES
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTHcalCalibTypeFilter.h,v 1.5 2009/08/27 13:33:46 gruen Exp $
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

class HLTHcalCalibTypeFilter : public HLTFilter {
public:
  explicit HLTHcalCalibTypeFilter(const edm::ParameterSet&);
  virtual ~HLTHcalCalibTypeFilter();
  
private:
  virtual void beginJob(void);
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
  virtual void endJob(void);
  
  // ----------member data ---------------------------
  
  edm::InputTag DataInputTag_ ;
  bool          Summary_ ;
  std::vector<int> CalibTypes_ ;   
  std::vector<int> eventsByType ; 

};

#endif
