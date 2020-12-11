#ifndef GeneratorInterface_LHEInterface_LH5Reader_h
#define GeneratorInterface_LHEInterface_LH5Reader_h

#include <string>
#include <vector>
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/LHEInterface/interface/lheh5.h"

namespace lhef {

  class LHERunInfo;
  class LHEEvent;

  class H5Handler {
  public:
    H5Handler(const std::string &fileNameIn);
    virtual ~H5Handler() {}
    void readBlock();
    void counter(int, int);
    std::unique_ptr<HighFive::File> h5file;
    bool indexStatus;
    HighFive::Group _index, _particle, _event, _init, _procInfo;
    int npLO, npNLO;
    unsigned int long _eventsRead;
    lheh5::Events2 _events2;
    lheh5::Events _events1;
    std::vector<lheh5::Particle> getEvent();
    lheh5::EventHeader getHeader();
    std::pair<lheh5::EventHeader, std::vector<lheh5::Particle> > getEventProperties();

  private:
    unsigned int long _eventsTotal;
    int _eventsInBlock;
    int _formatType;
    int _blocksRead;
  };

  class LH5Reader {
  public:
    LH5Reader(const edm::ParameterSet &params);
    LH5Reader(const std::vector<std::string> &fileNames, unsigned int skip = 0, int maxEvents = -1);
    LH5Reader(const std::string &inputs, unsigned int skip = 0, int maxEvents = -1);
    ~LH5Reader();

    std::shared_ptr<LHEEvent> next(bool *newFileOpened = nullptr);

  private:
    class Source;
    class FileSource;
    class StringSource;

    const std::vector<std::string> fileURLs;
    const std::string strName;
    unsigned int firstEvent;
    int maxEvents;
    unsigned int curIndex;
    std::vector<std::string> weightsinconfig;

    std::unique_ptr<Source> curSource;
    bool curDoc;
    std::shared_ptr<LHERunInfo> curRunInfo;
  };

}  // namespace lhef

#endif  // GeneratorInterface_LHEInterface_LH5Reader_h
