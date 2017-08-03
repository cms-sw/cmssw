#ifndef HLTriggerspecialHLTHcalLaserMisfireFilter_h
#define HLTriggerspecialHLTHcalLaserMisfireFilter_h
// -*- C++ -*-
//
// Package:    HLTHcalLaserMisfireFilter
// Class:      HLTHcalLaserMisfireFilter
// 
/**\class HLTHcalLaserMisfireFilter HLTHcalLaserMisfireFilter.cc filter/HLTHcalCalibTypeFilter/src/HLTHcalCalibTypeFilter.cc

Description: Filter to select HCAL Laser fired events

Implementation:
<Notes on implementation>
*/

// include files
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTHcalLaserMisfireFilter : public HLTFilter {
public:
  explicit HLTHcalLaserMisfireFilter(const edm::ParameterSet&);
  ~HLTHcalLaserMisfireFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:
 
  // ----------member data ---------------------------
  edm::InputTag                         inputHBHE_, inputHF_;
  edm::EDGetTokenT<HBHEDigiCollection>  inputTokenHBHE_;
  edm::EDGetTokenT<QIE10DigiCollection> inputTokenHF_;
  double                                minFracDiffHBHELaser_, minFracHFLaser_;
  int                                   minADCHBHE_, minADCHF_;
  bool                                  testMode_;
};

#endif
