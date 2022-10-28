#ifndef __L1Trigger_L1THGCal_HGCalAlgoWrapperBaseT_h__
#define __L1Trigger_L1THGCal_HGCalAlgoWrapperBaseT_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

template <typename InputCollection, typename OutputCollection, typename Tparam>
class HGCalAlgoWrapperBaseT {
public:
  HGCalAlgoWrapperBaseT(const edm::ParameterSet& conf) : name_(conf.getParameter<std::string>("AlgoName")) {}

  virtual ~HGCalAlgoWrapperBaseT() {}

  virtual void configure(const Tparam& parameters) = 0;
  virtual void process(const InputCollection& inputCollection, OutputCollection& outputCollection) const = 0;

  const std::string& name() const { return name_; }

private:
  const std::string name_;
};

#endif
