#ifndef GeneratorInterface_LHEInterface_LHEReader_h
#define GeneratorInterface_LHEInterface_LHEReader_h

#include <string>
#include <vector>
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {

  class XMLDocument;
  class LHERunInfo;
  class LHEEvent;

  class LHEReader {
  public:
    LHEReader(const edm::ParameterSet &params);
    LHEReader(const std::vector<std::string> &fileNames, unsigned int skip = 0);
    LHEReader(const std::string &inputs, unsigned int skip = 0);
    ~LHEReader();

    std::shared_ptr<LHEEvent> next(bool *newFileOpened = nullptr);

  private:
    class Source;
    class FileSource;
    class StringSource;
    class XMLHandler;

    const std::vector<std::string> fileURLs;
    const std::string strName;
    unsigned int firstEvent;
    int maxEvents;
    unsigned int curIndex;
    std::vector<std::string> weightsinconfig;

    std::unique_ptr<Source> curSource;
    std::unique_ptr<XMLDocument> curDoc;
    std::shared_ptr<LHERunInfo> curRunInfo;
    std::unique_ptr<XMLHandler> handler;
    std::shared_ptr<void> platform;
  };

}  // namespace lhef

#endif  // GeneratorInterface_LHEInterface_LHEReader_h
