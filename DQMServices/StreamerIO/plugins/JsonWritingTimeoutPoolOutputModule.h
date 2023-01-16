#ifndef DQMServices_StreamerIO_JsonWritingTimeoutPoolOutputModule_h
#define DQMServices_StreamerIO_JsonWritingTimeoutPoolOutputModule_h

#include "IOPool/Output/interface/TimeoutPoolOutputModule.h"

#include <string>
#include <utility>

namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;
}  // namespace edm

namespace dqmservices {

  class JsonWritingTimeoutPoolOutputModule : public edm::TimeoutPoolOutputModule {
  public:
    explicit JsonWritingTimeoutPoolOutputModule(edm::ParameterSet const& ps);
    ~JsonWritingTimeoutPoolOutputModule() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  protected:
    std::pair<std::string, std::string> physicalAndLogicalNameForNewFile() override;
    void doExtrasAfterCloseFile() override;

  protected:
    uint32_t const runNumber_;
    std::string const streamLabel_;
    std::string const outputPath_;

    uint32_t sequence_;
    std::string currentFileName_;
    std::string currentJsonName_;
  };

}  // namespace dqmservices

#endif  // DQMServices_StreamerIO_JsonWritingTimeoutPoolOutputModule_h
