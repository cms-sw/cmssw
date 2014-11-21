#ifndef IOPool_DQMStreamer_DQMFilerIterator_h
#define IOPool_DQMStreamer_DQMFilerIterator_h

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "boost/filesystem.hpp"

#include <map>
#include <queue>
#include <chrono>

#include "DQMMonitoringService.h"

namespace dqmservices {

class DQMFileIterator {
 public:
  struct LumiEntry {
    std::string filename;

    unsigned int file_ls;
    std::size_t n_events;
    std::string datafilename;

    static LumiEntry load_json(const std::string& filename, int lumiNumber,
                               unsigned int datafn_position);

    std::string state;
  };

  struct EorEntry {
    bool loaded = false;
    std::string filename;

    std::size_t n_events;
    std::size_t n_lumi;
    std::string datafilename;

    static EorEntry load_json(const std::string& filename);
  };

  enum State {
    OPEN = 0,
    EOR_CLOSING = 1,  // EoR file found, but lumis are still pending
    EOR = 2,
  };

  DQMFileIterator(edm::ParameterSet const& pset);
  ~DQMFileIterator();
  void initialise(int run, const std::string&, const std::string&);

  State state();

  /* methods to iterate the actual files */

  /* nextLumiNumber_ is the first unprocessed lumi number
   * lumiReady() returns if the next lumi is ready to be loaded
   * open() opens a file and advances the pointer to the next lumi
   *
   * front() a reference to the description (LumiEntry)
   * pop() advances to the next lumi
   */
  bool lumiReady();
  const LumiEntry open();

  void pop();
  std::string make_path_data(const LumiEntry& lumi);

  /* control */
  void reset();
  void update_state();

  /* misc helpers for input sources */
  void logFileAction(const std::string& msg,
                     const std::string& fileName = "") const;
  void logLumiState(const LumiEntry& lumi, const std::string& msg);

  void delay();
  void updateMonitoring();

  unsigned int runNumber();

  unsigned int lastLumiFound();
  void advanceToLumi(unsigned int lumi);

  static void fillDescription(edm::ParameterSetDescription& d);

 private:
  unsigned int runNumber_;
  std::string runInputDir_;
  std::string streamLabel_;
  unsigned long delayMillis_;
  long nextLumiTimeoutMillis_;
  long forceFileCheckTimeoutMillis_;

  // file name position in the json file
  unsigned int datafnPosition_;
  std::string runPath_;

  EorEntry eor_;
  State state_;

  unsigned int nextLumiNumber_;
  std::map<unsigned int, LumiEntry> lumiSeen_;

  /* this should be different,
   * since time between hosts might be not in sync */
  std::time_t runPathMTime_;
  std::chrono::high_resolution_clock::time_point runPathLastCollect_;

  /* this is for missing lumi files */
  std::chrono::high_resolution_clock::time_point lastLumiLoad_;

  void collect(bool ignoreTimers);

  /* this is for monitoring */
  edm::Service<DQMMonitoringService> mon_;
};

} /* end of namespace */

#endif
