#ifndef DQMServices_StreamerIO_DQMFileIterator_h
#define DQMServices_StreamerIO_DQMFileIterator_h

#include <chrono>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace dqmservices {

  class DQMMonitoringService;

  class DQMFileIterator {
  public:
    struct LumiEntry {
      std::string filename;
      std::string run_path;

      unsigned int file_ls;
      std::size_t n_events_processed;
      std::size_t n_events_accepted;
      std::string datafn;

      static LumiEntry load_json(const std::string& run_path,
                                 const std::string& filename,
                                 int lumiNumber,
                                 int datafn_position);

      std::string get_data_path() const;
      std::string get_json_path() const;
      std::string state;
    };

    struct EorEntry {
      bool loaded = false;
      std::string filename;
      std::string run_path;

      std::size_t n_events;
      std::size_t n_lumi;

      static EorEntry load_json(const std::string& run_path, const std::string& filename);
    };

    enum State {
      OPEN = 0,
      EOR_CLOSING = 1,  // EoR file found, but lumis are still pending
      EOR = 2,
    };

    DQMFileIterator(edm::ParameterSet const& pset);
    ~DQMFileIterator() = default;
    void initialise(int run, const std::string&, const std::string&);

    State state() const { return state_; }

    /* methods to iterate the actual files */

    /* nextLumiNumber_ is the first unprocessed lumi number
   * lumiReady() returns if the next lumi is ready to be loaded
   * open() opens a file and advances the pointer to the next lumi
   *
   * front() a reference to the description (LumiEntry)
   * pop() advances to the next lumi
   */
    bool lumiReady();
    LumiEntry open();

    void pop();

    /* control */
    void reset();
    void update_state();

    /* misc helpers for input sources */
    void logFileAction(const std::string& msg, const std::string& fileName = "") const;
    void logLumiState(const LumiEntry& lumi, const std::string& msg);

    void delay();

    unsigned int runNumber() const { return runNumber_; }
    unsigned int lastLumiFound();
    void advanceToLumi(unsigned int lumi, std::string reason);

    static void fillDescription(edm::ParameterSetDescription& d);

  private:
    unsigned int runNumber_;
    std::string runInputDir_;
    std::string streamLabel_;
    unsigned long delayMillis_;
    long nextLumiTimeoutMillis_;
    long forceFileCheckTimeoutMillis_;
    bool flagScanOnce_;

    // file name position in the json file
    unsigned int datafnPosition_;
    std::vector<std::string> runPath_;

    EorEntry eor_;
    State state_;

    unsigned int nextLumiNumber_;
    std::map<unsigned int, LumiEntry> lumiSeen_;
    std::unordered_set<std::string> filesSeen_;

    /* this should be different,
   * since time between hosts might be not in sync */
    unsigned runPathMTime_;
    std::chrono::high_resolution_clock::time_point runPathLastCollect_;

    /* this is for missing lumi files */
    std::chrono::high_resolution_clock::time_point lastLumiLoad_;

    unsigned mtimeHash() const;
    void collect(bool ignoreTimers);
    void monUpdateLumi(const LumiEntry& lumi);

    /* this is for monitoring */
    edm::Service<DQMMonitoringService> mon_;
  };

}  // namespace dqmservices

#endif  // DQMServices_StreamerIO_DQMFileIterator_h
