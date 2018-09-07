#ifndef RecoLocalFastTime_FTLCommonAlgos_MTDUncalibratedRecHitRecAlgoBase_HH
#define RecoLocalFastTime_FTLCommonAlgos_MTDUncalibratedRecHitRecAlgoBase_HH

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"

namespace edm {
  class Event;
  class EventSetup;
}

template <typename DataFrame> 
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
  virtual FTLUncalibratedRecHit makeRecHit(const DataFrame& dataFrame ) const = 0;

  const std::string& name() const { return name_; }

 private:
  std::string name_;

};


typedef MTDUncalibratedRecHitAlgoBase<BTLDataFrame> BTLUncalibratedRecHitAlgoBase;
typedef MTDUncalibratedRecHitAlgoBase<ETLDataFrame> ETLUncalibratedRecHitAlgoBase;


#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< BTLUncalibratedRecHitAlgoBase* (const edm::ParameterSet&, edm::ConsumesCollector&) > BTLUncalibratedRecHitAlgoFactory;
typedef edmplugin::PluginFactory< ETLUncalibratedRecHitAlgoBase* (const edm::ParameterSet&, edm::ConsumesCollector&) > ETLUncalibratedRecHitAlgoFactory;

#endif
