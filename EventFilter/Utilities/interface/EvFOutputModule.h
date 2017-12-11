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
    //pure in parent class but unused here
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override {}
    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {}
    void writeRun(edm::RunForOutput const&) override {}

    std::shared_ptr<EvFOutputJSONWriter> globalBeginRun(edm::Run const& run, edm::EventSetup const& setup) const override;
    std::shared_ptr<EvFOutputEventWriter> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override;
    void write(edm::EventForOutput const& e) override;
    void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override;

    Trig getTriggerResults(edm::EDGetTokenT<edm::TriggerResults> const& token, edm::EventForOutput const& e) const;

    edm::ParameterSet const& ps_;
    std::string streamLabel_;
    edm::EDGetTokenT<edm::TriggerResults> trToken_;

    evf::FastMonitoringService *fms_;

  }; //end-of-class-def

} // end of namespace-evf

#endif
