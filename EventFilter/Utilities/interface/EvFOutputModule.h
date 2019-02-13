#ifndef EventFilter_Utilities_EvFOutputModule_h
#define EventFilter_Utilities_EvFOutputModule_h

#include "FWCore/Framework/interface/limited/OutputModule.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleCommon.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

typedef edm::detail::TriggerResultsBasedEventSelector::handle_t Trig;

namespace evf {

  class FastMonitoringService;
  class EvFOutputEventWriter;
  class EvFOutputJSONWriter;

  typedef edm::limited::OutputModule<edm::LuminosityBlockCache<evf::EvFOutputEventWriter>,edm::RunCache<evf::EvFOutputJSONWriter>> EvFOutputModuleType;

  class EvFOutputModule : 
    public EvFOutputModuleType
  {
  public:
    explicit EvFOutputModule(edm::ParameterSet const& ps);
    ~EvFOutputModule() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
  private:
    void write(edm::EventForOutput const& e) override;

    //pure in parent class but unused here
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override {}
    void writeRun(edm::RunForOutput const&) override {}
    void globalEndRun(edm::RunForOutput const&) const override {}

    std::shared_ptr<EvFOutputJSONWriter> globalBeginRun(edm::RunForOutput const& run) const override;
    std::shared_ptr<EvFOutputEventWriter> globalBeginLuminosityBlock(edm::LuminosityBlockForOutput const& iLB) const override;
    void globalEndLuminosityBlock(edm::LuminosityBlockForOutput const& iLB) const override;

    Trig getTriggerResults(edm::EDGetTokenT<edm::TriggerResults> const& token, edm::EventForOutput const& e) const;

    edm::ParameterSet const& ps_;
    std::string streamLabel_;
    edm::EDGetTokenT<edm::TriggerResults> trToken_;

    evf::FastMonitoringService *fms_;

  }; //end-of-class-def

} // end of namespace-evf

#endif
