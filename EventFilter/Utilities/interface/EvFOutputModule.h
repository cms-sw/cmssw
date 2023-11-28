#ifndef EventFilter_Utilities_EvFOutputModule_h
#define EventFilter_Utilities_EvFOutputModule_h

#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleCommon.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#
typedef edm::detail::TriggerResultsBasedEventSelector::handle_t Trig;

namespace evf {

  class FastMonitoringService;
  //  class EvFOutputEventWriter;

  class EvFOutputEventWriter {
  public:
    explicit EvFOutputEventWriter(std::string const& filePath)
        : filePath_(filePath), accepted_(0), stream_writer_events_(new StreamerOutputFile(filePath)) {}

    ~EvFOutputEventWriter() {}

    void close() { stream_writer_events_->close(); }

    void doOutputEvent(EventMsgBuilder const& msg) {
      EventMsgView eview(msg.startAddress());
      stream_writer_events_->write(eview);
    }

    uint32 get_adler32() const { return stream_writer_events_->adler32(); }

    std::string const& getFilePath() const { return filePath_; }

    unsigned long getAccepted() const { return accepted_; }
    void incAccepted() { accepted_++; }

  private:
    std::string filePath_;
    unsigned long accepted_;
    edm::propagate_const<std::unique_ptr<StreamerOutputFile>> stream_writer_events_;
  };

  class EvFOutputJSONWriter {
  public:
    EvFOutputJSONWriter(edm::StreamerOutputModuleCommon::Parameters const& commonParameters,
                        edm::SelectedProducts const* selections,
                        std::string const& streamLabel,
                        std::string const& moduleLabel);

    edm::StreamerOutputModuleCommon streamerCommon_;

    jsoncollector::IntJ processed_;
    jsoncollector::IntJ accepted_;
    jsoncollector::IntJ errorEvents_;
    jsoncollector::IntJ retCodeMask_;
    jsoncollector::StringJ filelist_;
    jsoncollector::IntJ filesize_;
    jsoncollector::StringJ inputFiles_;
    jsoncollector::IntJ fileAdler32_;
    jsoncollector::StringJ transferDestination_;
    jsoncollector::StringJ mergeType_;
    jsoncollector::IntJ hltErrorEvents_;
    std::shared_ptr<jsoncollector::FastMonitor> jsonMonitor_;
    jsoncollector::DataPointDefinition outJsonDef_;
  };

  typedef edm::one::OutputModule<edm::one::WatchRuns, edm::LuminosityBlockCache<evf::EvFOutputEventWriter>>
      EvFOutputModuleType;

  class EvFOutputModule : public EvFOutputModuleType {
  public:
    explicit EvFOutputModule(edm::ParameterSet const& ps);
    ~EvFOutputModule() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void beginRun(edm::RunForOutput const& run) override;
    void write(edm::EventForOutput const& e) override;

    //pure in parent class but unused here
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override {}
    void writeRun(edm::RunForOutput const&) override {}
    void endRun(edm::RunForOutput const&) override {}

    std::shared_ptr<EvFOutputEventWriter> globalBeginLuminosityBlock(
        edm::LuminosityBlockForOutput const& iLB) const override;
    void globalEndLuminosityBlock(edm::LuminosityBlockForOutput const& iLB) override;

    Trig getTriggerResults(edm::EDGetTokenT<edm::TriggerResults> const& token, edm::EventForOutput const& e) const;

    edm::StreamerOutputModuleCommon::Parameters commonParameters_;
    std::string streamLabel_;
    edm::EDGetTokenT<edm::TriggerResults> trToken_;
    edm::EDGetTokenT<edm::SendJobHeader::ParameterSetMap> psetToken_;

    evf::FastMonitoringService* fms_;

    std::unique_ptr<evf::EvFOutputJSONWriter> jsonWriter_;

  };  //end-of-class-def

}  // namespace evf

#endif
