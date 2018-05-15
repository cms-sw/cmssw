#ifndef RecoLocalFastTime_FTLCommonAlgos_MTDUncalibratedRecHitRecAlgoBase_HH
#define RecoLocalFastTime_FTLCommonAlgos_MTDUncalibratedRecHitRecAlgoBase_HH

/** \class MTDUncalibRecHitRecAlgoBase
  *  Template used by Ecal to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  \author
  */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"

namespace edm {
  class Event;
  class EventSetup;
}

class MTDUncalibratedRecHitAlgoBase {
 public:
  /// Constructor
  MTDUncalibratedRecHitAlgoBase(const edm::ParameterSet& conf,
                                edm::ConsumesCollector& sumes) { }

  /// Destructor
  virtual ~MTDUncalibratedRecHitAlgoBase() { }

  /// get event and eventsetup information
  virtual void getEvent(const edm::Event&) = 0;
  virtual void getEventSetup(const edm::EventSetup&) = 0;

  /// make the rec hit
  
  virtual FTLUncalibratedRecHit makeRecHit(const BTLDataFrame& dataFrame ) const = 0;
  virtual FTLUncalibratedRecHit makeRecHit(const ETLDataFrame& dataFrame ) const = 0;

  const std::string& name() const { return name_; }

 private:
  std::string name_;

};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< MTDUncalibratedRecHitAlgoBase* (const edm::ParameterSet&, edm::ConsumesCollector&) > MTDUncalibratedRecHitAlgoFactory;


#endif
