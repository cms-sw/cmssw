#ifndef IOPool_DQMStreamer_DQMFilerIterator_h
#define IOPool_DQMStreamer_DQMFilerIterator_h

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "boost/filesystem.hpp"

#include <memory>
#include <map>
#include <string>
#include <queue>
#include <iterator>
#include <chrono>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace edm {

class DQMFileIterator {
 public:
  enum JsonType {
    JS_PROTOBUF,
    JS_DATA,
  };

  struct LumiEntry {
    bool loaded = false;
    std::string filename;

    int ls;
    std::size_t n_events;
    std::string datafilename;

    static LumiEntry load_json(const std::string& filename, int lumiNumber,
                               JsonType type);
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

  DQMFileIterator(ParameterSet const& pset, JsonType t);
  ~DQMFileIterator();
  void initialise(int run, const std::string&, const std::string&);

  State state();

  /* methods to iterate the actual files */

  /* currentLumi_ is the first unprocessed lumi number
   * lumiReady() returns if it is loadable
   * front() a reference to the description (LumiEntry)
   * pop() advances to the next lumi
   */
  bool lumiReady();
  const LumiEntry& front();
  void pop();
  std::string make_path_data(const LumiEntry& lumi);

  /* control */
  void reset();
  void update_state();

  /* misc helpers for input sources */
  void logFileAction(const std::string& msg,
                     const std::string& fileName = "") const;
  void delay();
  void updateWatchdog();
  unsigned int runNumber();

  unsigned int lastLumiFound();
  void advanceToLumi(unsigned int lumi);

  static void fillDescription(ParameterSetDescription& d);

 private:
  JsonType type_;

  unsigned int runNumber_;
  std::string runInputDir_;
  std::string streamLabel_;
  unsigned long delayMillis_;
  long nextLumiTimeoutMillis_;

  std::string runPath_;

  EorEntry eor_;
  State state_;

  unsigned int currentLumi_;
  std::map<unsigned int, LumiEntry> lumiSeen_;

  /* this should be different,
   * since time between hosts might be not in sync */
  std::time_t runPathMTime_;
  std::chrono::high_resolution_clock::time_point runPathLastCollect_;

  /* this is for missing lumi files */
  std::chrono::high_resolution_clock::time_point lastLumiLoad_;

  void collect(bool ignoreTimers);
};

} /* end of namespace */
#endif
