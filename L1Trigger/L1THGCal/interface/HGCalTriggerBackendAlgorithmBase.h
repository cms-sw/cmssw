#ifndef __L1Trigger_L1THGCal_HGCalTriggerBackendAlgorithmBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerBackendAlgorithmBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiDefs.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include <memory>

/*******
 *
 * class: HGCalTriggerBackendAlgorithmBase
 * author: L.Gray (FNAL)
 * date: 27 July, 2015
 *
 * This contains two base classes, one which presents the abstract interface,
 * and a second that allows specialization to various FE codecs so that 
 * data can be manipulated directly.
 *
 *******/

class HGCalTriggerBackendAlgorithmBase {
public:
  // Allow HGCalTriggerBackend to be passed a consume collector
  HGCalTriggerBackendAlgorithmBase(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : geometry_(nullptr), name_(conf.getParameter<std::string>("AlgorithmName")) {}

  virtual ~HGCalTriggerBackendAlgorithmBase() {}

  const std::string& name() const { return name_; }

  virtual void setGeometry(const HGCalTriggerGeometryBase* const geom) { geometry_ = geom; }

  //runs the trigger algorithm, storing internally the results
  virtual void setProduces(edm::stream::EDProducer<>& prod) const = 0;

  virtual void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es, edm::Event& e) = 0;

  virtual void putInEvent(edm::Event& evt) = 0;

  virtual void reset() = 0;

protected:
  const HGCalTriggerGeometryBase* geometry_;

private:
  const std::string name_;
};

// ----> all backend algorithm classes derive from this <----
// inheritance looks like class MyAlgorithm : public HGCalTriggerBackend::Algorithm<MyCodec>
namespace HGCalTriggerBackend {
  template <typename FECODEC>
  class Algorithm : public HGCalTriggerBackendAlgorithmBase {
  public:
    Algorithm(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
        : HGCalTriggerBackendAlgorithmBase(conf, cc), codec_(conf.getParameterSet("FECodec")) {}

    void setGeometry(const HGCalTriggerGeometryBase* const geom) final {
      HGCalTriggerBackendAlgorithmBase::setGeometry(geom);
      codec_.setGeometry(geom);
    }

  protected:
    FECODEC codec_;
  };
}  // namespace HGCalTriggerBackend

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalTriggerBackendAlgorithmBase*(const edm::ParameterSet&, edm::ConsumesCollector&)>
    HGCalTriggerBackendAlgorithmFactory;

#endif
