#ifndef RecoLocalFastTime_FTLCommonAlgos_FTLRecHitAlgoBase_HH
#define RecoLocalFastTime_FTLCommonAlgos_FTLRecHitAlgoBase_HH

/** \class FTLRecHitAlgoBase
  *  Template algorithm to make rechits from uncalibrated rechits
  *
  *  \author Lindsey Gray
  */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"

namespace edm {
  class Event;
  class EventSetup;
}

class FTLRecHitAlgoBase {
 public:

  /// Constructor
  FTLRecHitAlgoBase(const edm::ParameterSet& conf,
                    edm::ConsumesCollector& sumes):
    name_( conf.getParameter<std::string>("algoName") ){ };

  /// Destructor
  virtual ~FTLRecHitAlgoBase() { };

  /// get event and eventsetup information
  virtual void getEvent(const edm::Event&) = 0;
  virtual void getEventSetup(const edm::EventSetup&) = 0;

  /// make rechits from dataframes
  virtual FTLRecHit makeRecHit(const FTLUncalibratedRecHit& uncalibRH, uint32_t& flags) const = 0;

  const std::string& name() const { return name_; }

 private:
  std::string name_;

};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< FTLRecHitAlgoBase* (const edm::ParameterSet&, edm::ConsumesCollector&) > FTLRecHitAlgoFactory;


#endif
