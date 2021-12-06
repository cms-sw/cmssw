#ifndef __L1Trigger_L1THGCal_HGCalProcessorBaseT_h__
#define __L1Trigger_L1THGCal_HGCalProcessorBaseT_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

template <typename InputCollection, typename OutputCollection>
class HGCalProcessorBaseT {
public:
  HGCalProcessorBaseT(const edm::ParameterSet& conf)
      : geometry_(nullptr), name_(conf.getParameter<std::string>("ProcessorName")) {}

  virtual ~HGCalProcessorBaseT() {}

  const std::string& name() const { return name_; }

  virtual void setGeometry(const HGCalTriggerGeometryBase* const geom) { geometry_ = geom; }

  virtual void run(const InputCollection& inputColl, OutputCollection& outColl) = 0;

protected:
  const HGCalTriggerGeometryBase* geometry() const {
    if (!geometry_)
      throw cms::Exception("HGCTriggerSetupError") << "The geometry has not been set in processor " << name_;
    return geometry_;
  }

private:
  const HGCalTriggerGeometryBase* geometry_;
  const std::string name_;
};

#endif
