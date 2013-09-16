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
//
//


// include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTHcalCalibTypeFilter : public edm::EDFilter {
public:
  explicit HLTHcalCalibTypeFilter(const edm::ParameterSet&);
  virtual ~HLTHcalCalibTypeFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:
  virtual void beginJob(void);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob(void);
  
  // ----------member data ---------------------------
  
  edm::EDGetTokenT<FEDRawDataCollection> DataInputToken_;
  edm::InputTag DataInputTag_ ;
  bool          Summary_ ;
  std::vector<int> CalibTypes_ ;   
  std::vector<int> eventsByType ; 

};

#endif
