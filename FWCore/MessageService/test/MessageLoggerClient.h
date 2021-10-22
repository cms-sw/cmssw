#ifndef Integration_MessageLoggerClient_h
#define Integration_MessageLoggerClient_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class MessageLoggerClient : public edm::global::EDAnalyzer<> {
  public:
    explicit MessageLoggerClient(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const final;

  private:
  };

}  // namespace edmtest

#endif  // Integration_MessageLoggerClient_h
