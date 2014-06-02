#ifndef IOPool_DQMStreamer_DQMFilerIterator_h
#define IOPool_DQMStreamer_DQMFilerIterator_h

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "boost/filesystem.hpp"

#include <memory>
#include <string>
#include <queue>
#include <iterator>
#include <chrono>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace edm {

class DQMFileIterator {
 public:
  struct LumiEntry {
    int ls;

    std::size_t n_events;
    std::string datafilename;
    std::string definition;
    std::string source;

    static LumiEntry load_json(const std::string& filename, int lumiNumber);
  };

  struct EorEntry {
    bool loaded = false;

    std::size_t n_events;
    std::size_t n_lumi;
    std::string datafilename;
    std::string definition;
    std::string source;

    static EorEntry load_json(const std::string& filename);
  };

  enum State {
    OPEN = 0,
    EOR_CLOSING = 1,  // EoR file found, but lumis are still pending
    EOR = 2,
  };

  DQMFileIterator(ParameterSet const& pset);
  ~DQMFileIterator();
  void initialise(int run, const std::string&, const std::string&);

  State state();

  /* methods to iterate the actual files */
  const LumiEntry& front();
  void pop();
  bool hasNext();

  std::string make_path_jsn(int lumi);
  std::string make_path_eor();
  std::string make_path_data(const LumiEntry& lumi);

  /* control */
  void reset();
  void collect();
  void update_state();

  /* misc helpers for input sources */
  void logFileAction(const std::string& msg, const std::string& fileName="") const;
  void delay();
  void updateWatchdog();
  unsigned int runNumber() { return runNumber_; };

  static void fillDescription(ParameterSetDescription& d);

 private:
  unsigned int runNumber_;
  std::string runInputDir_;
  std::string streamLabel_;
  unsigned int delayMillis_;

  std::string runPath_;

  int lastLumiSeen_;
  EorEntry eor_;
  State state_;
  std::queue<LumiEntry> queue_;

  std::chrono::high_resolution_clock::time_point last_collect_;
};

} /* end of namespace */
#endif
