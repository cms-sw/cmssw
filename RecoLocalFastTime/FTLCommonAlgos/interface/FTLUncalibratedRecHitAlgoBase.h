#ifndef RecoLocalFastTime_FTLCommonAlgos_FTLUncalibratedRecHitRecAlgoBase_HH
#define RecoLocalFastTime_FTLCommonAlgos_FTLUncalibratedRecHitRecAlgoBase_HH

/** \class FTLUncalibRecHitRecAlgoBase
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

class FTLUncalibratedRecHitAlgoBase {
 public:
  /// Constructor
  FTLUncalibratedRecHitAlgoBase(const edm::ParameterSet& conf,
                                edm::ConsumesCollector& sumes) { }

  /// Destructor
  virtual ~FTLUncalibratedRecHitAlgoBase() { }

  /// get event and eventsetup information
  virtual void getEvent(const edm::Event&) = 0;
  virtual void getEventSetup(const edm::EventSetup&) = 0;

  /// make the rec hit
  virtual FTLUncalibratedRecHit makeRecHit(const FTLDataFrame& dataFrame ) const = 0;

  const std::string& name() const { return name_; }

 private:
  std::string name_;

};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< FTLUncalibratedRecHitAlgoBase* (const edm::ParameterSet&, edm::ConsumesCollector&) > FTLUncalibratedRecHitAlgoFactory;


#endif
