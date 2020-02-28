#ifndef PhysicsTools_PatAlgos_interface_SimpleIsolator_h
#define PhysicsTools_PatAlgos_interface_SimpleIsolator_h

#include "PhysicsTools/PatAlgos/interface/BaseIsolator.h"

namespace pat {
  namespace helper {
    class SimpleIsolator : public BaseIsolator {
    public:
      typedef edm::ValueMap<double> IsoValueMap;
      SimpleIsolator() {}
      SimpleIsolator(const edm::ParameterSet &conf, edm::ConsumesCollector &iC, bool withCut);
      ~SimpleIsolator() override {}
      void beginEvent(const edm::Event &event, const edm::EventSetup &eventSetup) override;
      void endEvent() override;

      std::string description() const override { return input_.encode(); }

    protected:
      edm::Handle<IsoValueMap> handle_;
      edm::EDGetTokenT<IsoValueMap> inputDoubleToken_;
      float getValue(const edm::ProductID &id, size_t index) const override { return handle_->get(id, index); }
    };  // class SimpleIsolator
  }     // namespace helper
}  // namespace pat

#endif
