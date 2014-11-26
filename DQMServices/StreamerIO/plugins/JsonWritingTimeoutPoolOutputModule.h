#ifndef DQMServices_StreamerIO_JsonWritingTimeoutPoolOutputModule_h
#define DQMServices_StreamerIO_JsonWritingTimeoutPoolOutputModule_h

#include "IOPool/Output/interface/TimeoutPoolOutputModule.h"
#include "IOPool/Output/interface/RootOutputFile.h"

namespace dqmservices {

class ModuleCallingContext;
class ParameterSet;

class JsonWritingTimeoutPoolOutputModule : public edm::TimeoutPoolOutputModule {
 public:
  explicit JsonWritingTimeoutPoolOutputModule(edm::ParameterSet const& ps);
  virtual ~JsonWritingTimeoutPoolOutputModule(){};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

 protected:
  virtual void reallyOpenFile() override;
  virtual void reallyCloseFile() override;

 protected:
  uint32_t sequence_;
  uint32_t runNumber_;
  std::string streamLabel_;
  std::string outputPath_;

  std::string currentFileName_;
  std::string currentJsonName_;
};

}  // end of namespace

#endif
