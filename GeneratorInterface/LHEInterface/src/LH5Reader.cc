#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include "GeneratorInterface/LHEInterface/interface/LH5Reader.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
//#include "GeneratorInterface/LHEInterface/code/HighFive/include/highfive/H5File.hpp"
#include "GeneratorInterface/LHEInterface/interface/lheh5.h"

#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

namespace lhef {
  using namespace lheh5;

  static void logFileAction(char const *msg, std::string const &fileName) {
    edm::LogAbsolute("fileAction") << std::setprecision(0) << edm::TimeOfDay() << msg << fileName;
    edm::FlushMessageLog();
  }

  class LH5Reader::Source {
  public:
    Source() {}
    virtual ~Source() {}
    std::unique_ptr<H5Handler> handler;

  private:
  };

  class LH5Reader::FileSource : public LH5Reader::Source {
  public:
    FileSource(const std::string &fileURL) {
      const std::string fileName = fileURL.substr(5, fileURL.length());

      H5Handler *tmph = new H5Handler(fileName);
      handler.reset(tmph);
    }

    ~FileSource() override {}
  };

  class LH5Reader::StringSource : public LH5Reader::Source {
  public:
    StringSource(const std::string &inputs) {
      if (inputs.empty())
        throw cms::Exception("StreamOpenError") << "Empty LHE file string name \"" << std::endl;

      H5Handler *tmph = new H5Handler(inputs);
      handler.reset(tmph);
    }

    ~StringSource() override {}
  };

  H5Handler::H5Handler(const std::string &fileNameIn)
      : h5file(new HighFive::File(fileNameIn)),
        indexStatus(h5file->exist("/index")),
        _index(h5file->getGroup(indexStatus ? "index" : "event")),
        _particle(h5file->getGroup("particle")),
        _event(h5file->getGroup("event")),
        _init(h5file->getGroup("init")),
        _procInfo(h5file->getGroup("procInfo")) {
    hid_t dspace;
    _formatType = 1;

    if (indexStatus) {
      dspace = H5Dget_space(h5file->getDataSet("index/start").getId());
      _formatType = 2;
    } else {
      _index = h5file->getGroup("event");
      dspace = H5Dget_space(h5file->getDataSet("event/start").getId());
      _formatType = 1;
    }

    _eventsTotal = H5Sget_simple_extent_npoints(dspace);
    _eventsRead = 0;
    _eventsInBlock = 6400 / 4;  // configurable parameter??
    if (_eventsTotal % _eventsInBlock != 0)
      throw cms::Exception("ReadError") << "block size does not match HDF5 file" << std::endl;
    _blocksRead = 0;

    // Check if the file contains the npLO, npNLO information
    bool status = h5file->exist("/procInfo/npLO");
    if (status) {
      HighFive::DataSet _npLO = _procInfo.getDataSet("npLO");
      _npLO.read(npLO);
      HighFive::DataSet _npNLO = _procInfo.getDataSet("npNLO");
      _npNLO.read(npNLO);
    } else {
      npLO = 1;
      npNLO = 0;
    }
  };

  void H5Handler::counter(const int firstEventIn, const int maxEventsIn) {
    if (maxEventsIn > 0 && firstEventIn > maxEventsIn)
      throw cms::Exception("ConfigurationError") << "\" firstEvent > maxEvents \"" << std::endl;
    // Reset maximum number of events to read
    if (_blocksRead == 0 && maxEventsIn >= 0 && (unsigned long int)maxEventsIn < _eventsTotal)
      _eventsTotal = (unsigned long int)maxEventsIn;
    // If there are multiple files, assume you only want to jump through the first file
    if (firstEventIn > 0 && _blocksRead > 0)
      _eventsRead = firstEventIn - 1;
    // Set blocks read to be in the correct place
    _blocksRead = _eventsRead / _eventsInBlock;
  }

  void H5Handler::readBlock() {
    // Start counting at 0
    size_t iStart = _blocksRead * _eventsInBlock;
    size_t nEvents = _eventsInBlock;
    if ((unsigned long int)(iStart + nEvents) > _eventsTotal)
      return;
    _events1 = readEvents(_index, _particle, _event, iStart, nEvents);
    _blocksRead++;
  }

  std::vector<lheh5::Particle> H5Handler::getEvent() {
    std::vector<lheh5::Particle> _vE;
    if (_eventsRead > _eventsTotal - 1)
      return std::vector<lheh5::Particle>();
    int checkEvents = (_blocksRead)*_eventsInBlock - 1;
    _vE = _events1.mkEvent(_eventsRead);
    _eventsRead++;
    if (checkEvents >= 0 && _eventsRead > (unsigned long int)checkEvents)
      readBlock();
    return _vE;
  }

  lheh5::EventHeader H5Handler::getHeader() {
    // fragile, must be called before getEvent
    return _events1.mkEventHeader(_eventsRead);
  }

  std::pair<lheh5::EventHeader, std::vector<lheh5::Particle> > H5Handler::getEventProperties() {
    std::vector<lheh5::Particle> _vE;
    lheh5::EventHeader _h;
    _h.nparticles = -1;
    if (_eventsRead > _eventsTotal - 1)
      return std::make_pair(_h, _vE);
    int checkEvents = (_blocksRead)*_eventsInBlock - 1;
    _vE = _events1.mkEvent(_eventsRead % _eventsInBlock);
    _h = _events1.mkEventHeader(_eventsRead % _eventsInBlock);
    _eventsRead++;
    if (checkEvents >= 0 && _eventsRead > (unsigned long int)checkEvents)
      readBlock();
    return std::make_pair(_h, _vE);
  }

  LH5Reader::LH5Reader(const edm::ParameterSet &params)
      : fileURLs(params.getUntrackedParameter<std::vector<std::string> >("fileNames")),
        strName(""),
        firstEvent(params.getUntrackedParameter<unsigned int>("skipEvents", 0)),
        maxEvents(params.getUntrackedParameter<int>("limitEvents", -1)) {}

  LH5Reader::LH5Reader(const std::vector<std::string> &fileNames, unsigned int firstEvent, int maxEvents)
      : fileURLs(fileNames), strName(""), firstEvent(firstEvent), maxEvents(maxEvents), curIndex(0), curDoc(false) {}

  LH5Reader::LH5Reader(const std::string &inputs, unsigned int firstEvent, int maxEvents)
      : strName(inputs), firstEvent(firstEvent), maxEvents(maxEvents), curIndex(0) {}

  LH5Reader::~LH5Reader() { curSource.release(); }

  std::shared_ptr<LHEEvent> LH5Reader::next(bool *newFileOpened) {
    while (curDoc || curIndex < fileURLs.size() || (fileURLs.empty() && !strName.empty())) {
      if (!curDoc) {
        if (!fileURLs.empty()) {
          logFileAction("  Initiating request to open LHE file ", fileURLs[curIndex]);
          curSource = std::make_unique<FileSource>(fileURLs[curIndex]);
          logFileAction("  Successfully opened LHE file ", fileURLs[curIndex]);
          if (newFileOpened != nullptr)
            *newFileOpened = true;
          ++curIndex;
        } else if (!strName.empty()) {
          curSource = std::make_unique<StringSource>(strName);
        }
        // New "doc" has been opened.    This is the same as a new source.
        curDoc = true;
        // Set maxEvents and firstEvent
        curSource->handler->counter(firstEvent, maxEvents);
        curSource->handler->readBlock();

        curRunInfo.reset();
        HEPRUP tmprup;
        int beamA, beamB;
        curSource->handler->_init.getDataSet("beamA").read(beamA);
        curSource->handler->_init.getDataSet("beamB").read(beamB);
        tmprup.IDBMUP = std::make_pair(beamA, beamB);
        double energyA, energyB;
        curSource->handler->_init.getDataSet("energyA").read(energyA);
        curSource->handler->_init.getDataSet("energyB").read(energyB);
        tmprup.EBMUP = std::make_pair(energyA, energyB);
        int PDFsetA, PDFsetB;
        curSource->handler->_init.getDataSet("PDFsetA").read(PDFsetA);
        curSource->handler->_init.getDataSet("PDFsetB").read(PDFsetB);
        tmprup.PDFSUP = std::make_pair(PDFsetA, PDFsetB);
        int PDFgroupA, PDFgroupB;
        curSource->handler->_init.getDataSet("PDFgroupA").read(PDFgroupA);
        curSource->handler->_init.getDataSet("PDFgroupB").read(PDFgroupB);
        tmprup.PDFGUP = std::make_pair(PDFgroupA, PDFgroupB);
        std::vector<int> procId;         // NOTE: C++17 allows int[numProcesses]
        std::vector<double> xSection;    // NOTE: C++17 allows double[numProcesses]
        std::vector<double> error;       // NOTE: C++17 allows double[numProcesses]
        std::vector<double> unitWeight;  // NOTE: C++17 allows double[numProcesses]

        curSource->handler->_procInfo.getDataSet("procId").read(procId);
        curSource->handler->_procInfo.getDataSet("xSection").read(xSection);
        curSource->handler->_procInfo.getDataSet("error").read(error);
        curSource->handler->_procInfo.getDataSet("unitWeight").read(unitWeight);

        tmprup.LPRUP = procId;
        tmprup.XSECUP = xSection;
        tmprup.XERRUP = error;
        tmprup.XMAXUP = unitWeight;
        tmprup.IDWTUP = 3;
        size_t numProcesses = procId.size();
        tmprup.NPRUP = numProcesses;
        // Use temporary process info block to define const HEPRUP
        const HEPRUP heprup(tmprup);

        curRunInfo.reset(new LHERunInfo(heprup));
        // Run info has now been set when a new file is encountered
      }
      // Handler should be modified to have these capabilities
      // Maybe this is set event by event??
      int npLO = curSource->handler->npLO;
      int npNLO = curSource->handler->npNLO;

      // Event-loop here
      std::pair<EventHeader, std::vector<Particle> > evp = curSource->handler->getEventProperties();
      EventHeader hd = evp.first;
      if (hd.nparticles < 0) {
        curDoc = false;
        logFileAction("  Closed LHE file ", fileURLs[curIndex - 1]);
        return std::shared_ptr<LHEEvent>();
      }
      HEPEUP tmp;
      tmp.resize(hd.nparticles);
      //Particle loop
      unsigned int ip = 0;
      //      for (auto part: curSource->handler->_events1.mkEvent(i)) {
      for (auto part : evp.second) {
        tmp.IDUP[ip] = part.id;
        tmp.ISTUP[ip] = part.status;
        tmp.MOTHUP[ip] = std::make_pair(part.mother1, part.mother2);
        tmp.ICOLUP[ip] = std::make_pair(part.color1, part.color2);
        tmp.VTIMUP[ip] = part.lifetime;
        tmp.SPINUP[ip] = part.spin;
        tmp.PUP[ip][0] = part.px;
        tmp.PUP[ip][1] = part.py;
        tmp.PUP[ip][2] = part.pz;
        tmp.PUP[ip][3] = part.e;
        tmp.PUP[ip][4] = part.m;
        ip++;
      }
      tmp.IDPRUP = hd.pid;
      tmp.XWGTUP = hd.weight;
      tmp.SCALUP = hd.scale;
      tmp.AQEDUP = hd.aqed;
      tmp.AQCDUP = hd.aqcd;
      std::shared_ptr<LHEEvent> lheevent;
      // Use temporary event to construct const HEPEUP;
      const HEPEUP hepeup(tmp);

      lheevent.reset(new LHEEvent(curRunInfo, hepeup));
      // Might have to add this capability later
      /*          const XMLHandler::wgt_info &info = handler->weightsinevent;
          for (size_t i = 0; i < info.size(); ++i) {
            double num = -1.0;
            sscanf(info[i].second.c_str(), "%le", &num);
            lheevent->addWeight(gen::WeightsInfo(info[i].first, num));
            }*/
      // Currently these are set just at the beginning?
      // might be an event property
      lheevent->setNpLO(npLO);
      lheevent->setNpNLO(npNLO);
      //fill scales
      /*          if (!handler->scales.empty()) {
            lheevent->setScales(handler->scales);
            }*/
      return lheevent;
    }
    return std::shared_ptr<LHEEvent>();
  }

}  // namespace lhef
