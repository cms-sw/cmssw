#include "DQMFileIterator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/range.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <memory>
#include <string>
#include <iterator>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace dqmservices {

DQMFileIterator::LumiEntry DQMFileIterator::LumiEntry::load_json(
    const std::string& filename, int lumiNumber, unsigned int datafn_position) {
  boost::property_tree::ptree pt;
  read_json(filename, pt);

  LumiEntry lumi;
  lumi.filename = filename;

  // We rely on n_events to be the first item on the array...
  lumi.n_events = std::next(pt.get_child("data").begin(), 1)
                      ->second.get_value<std::size_t>();

  lumi.ls = lumiNumber;
  lumi.datafilename = std::next(pt.get_child("data").begin(), datafn_position)
    ->second.get_value<std::string>();

  lumi.loaded = true;
  return lumi;
}

// Contents of Eor json file are ignored for the moment.
// This function will not be called.
DQMFileIterator::EorEntry DQMFileIterator::EorEntry::load_json(
    const std::string& filename) {
  boost::property_tree::ptree pt;
  read_json(filename, pt);

  EorEntry eor;
  eor.filename = filename;

  // We rely on n_events to be the first item on the array...
  eor.n_events = std::next(pt.get_child("data").begin(), 1)
                     ->second.get_value<std::size_t>();
  eor.n_lumi = std::next(pt.get_child("data").begin(), 2)
                   ->second.get_value<std::size_t>();
  eor.datafilename = std::next(pt.get_child("data").begin(), 2)
                         ->second.get_value<std::string>();

  eor.loaded = true;
  return eor;
}

DQMFileIterator::DQMFileIterator(edm::ParameterSet const& pset)
    : state_(EOR) {

  runNumber_ = pset.getUntrackedParameter<unsigned int>("runNumber");
  datafnPosition_ = pset.getUntrackedParameter<unsigned int>("datafnPosition");
  runInputDir_ = pset.getUntrackedParameter<std::string>("runInputDir");
  streamLabel_ = pset.getUntrackedParameter<std::string>("streamLabel");
  delayMillis_ = pset.getUntrackedParameter<uint32_t>("delayMillis");
  nextLumiTimeoutMillis_ =
      pset.getUntrackedParameter<int32_t>("nextLumiTimeoutMillis");

  reset();
}

DQMFileIterator::~DQMFileIterator() {}

void DQMFileIterator::reset() {
  runPath_ = str(boost::format("%s/run%06d") % runInputDir_ % runNumber_);

  eor_.loaded = false;
  state_ = State::OPEN;
  currentLumi_ = 1;
  lumiSeen_.clear();

  lastLumiLoad_ = std::chrono::high_resolution_clock::now();

  collect(true);
  update_state();
}

DQMFileIterator::State DQMFileIterator::state() { return state_; }

const DQMFileIterator::LumiEntry& DQMFileIterator::front() {
  return lumiSeen_[currentLumi_];
}

void DQMFileIterator::pop() {
  advanceToLumi(currentLumi_ + 1);
}

bool DQMFileIterator::lumiReady() {
  if (lumiSeen_.find(currentLumi_) != lumiSeen_.end()) {
    return true;
  }

  return false;
}

unsigned int DQMFileIterator::runNumber() { return runNumber_; }

unsigned int DQMFileIterator::lastLumiFound() {
  if (!lumiSeen_.empty()) {
    return lumiSeen_.rbegin()->first;
  }

  return 1;
}

void DQMFileIterator::advanceToLumi(unsigned int lumi) {
  using boost::property_tree::ptree;
  using boost::str;

  unsigned int prev_lumi = currentLumi_;

  currentLumi_ = lumi;
  lastLumiLoad_ = std::chrono::high_resolution_clock::now();

  // report the successful lumi file open
  if (mon_.isAvailable()) {
    ptree children;

    auto iter = lumiSeen_.begin();
    for (; iter != lumiSeen_.end(); ++iter) {
      children.put(std::to_string(iter->first), iter->second.filename);
    }

    mon_->registerExtra("lumiSeen", children);
    mon_->reportLumiSection(runNumber_, prev_lumi);
  }
}

std::string DQMFileIterator::make_path_data(const LumiEntry& lumi) {
  if (boost::starts_with(lumi.datafilename, "/")) return lumi.datafilename;

  boost::filesystem::path p(runPath_);
  p /= lumi.datafilename;
  return p.string();
}

void DQMFileIterator::collect(bool ignoreTimers) {
  // search filesystem to find available lumi section files
  // or the end of run files

  auto now = std::chrono::high_resolution_clock::now();
  auto last_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - runPathLastCollect_).count();

  // don't refresh if it's too soon
  if ((!ignoreTimers) && (last_ms < 100)) {
    return;
  } else {
    runPathLastCollect_ = now;
  }

  // check if directory changed
  std::time_t t = boost::filesystem::last_write_time(runPath_);

  if ((!ignoreTimers) && (t <= runPathMTime_)) {
    //logFileAction("Directory hasn't changed.");
    return;
  } else {
    //logFileAction("Directory changed, updating.");
    runPathMTime_ = t;
  }

  using boost::filesystem::directory_iterator;
  using boost::filesystem::directory_entry;

  std::string fn_eor;

  directory_iterator dend;
  for (directory_iterator di(runPath_); di != dend; ++di) {
    const boost::regex fn_re("run(\\d+)_ls(\\d+)(_.*).jsn");

    const std::string filename = di->path().filename().string();
    const std::string fn = di->path().string();

    boost::smatch result;
    if (boost::regex_match(filename, result, fn_re)) {
      unsigned int run = std::stoi(result[1]);
      unsigned int lumi = std::stoi(result[2]);
      std::string label = result[3];

      if (run != runNumber_) continue;

      // check if this is EoR
      // for various reasons we have to load it after all other files
      if ((lumi == 0) && (label == "_EoR") && (!eor_.loaded)) {
        fn_eor = fn;
        continue;
      }

      // check if lumi is loaded
      if (lumiSeen_.find(lumi) != lumiSeen_.end()) {
        continue;  // already loaded
      }

      // check if this belongs to us
      if (label != streamLabel_) {
        logFileAction("Found and skipped json file (stream label mismatch): ",
                      fn);
        continue;
      }

      LumiEntry lumi_jsn = LumiEntry::load_json(fn, lumi, datafnPosition_);
      lumiSeen_.emplace(lumi, lumi_jsn);
      logFileAction("Found and loaded json file: ", fn);
    }
  }

  if (!fn_eor.empty()) {
    logFileAction("EoR file found: ", fn_eor);

    // @TODO load EoR files correctly
    // eor_ = EorEntry::load_json(fn_eor);
    // logFileAction("Loaded eor file: ", fn_eor);

    // for now , set n_lumi to the highest _found_ lumi
    eor_.loaded = true;

    if (lumiSeen_.empty()) {
      eor_.n_lumi = 0;
    } else {
      eor_.n_lumi = lumiSeen_.rbegin()->first;
    }
  }
}

void DQMFileIterator::update_state() {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::milliseconds;

  collect(false);

  // now update the state
  State old_state = state_;

  if ((state_ == State::OPEN) && (eor_.loaded)) {
    state_ = State::EOR_CLOSING;
  }

  // special case for missing lumi files
  // skip to the next available, but after the timeout
  if ((state_ != State::EOR) && (nextLumiTimeoutMillis_ >= 0)) {
    auto iter = lumiSeen_.lower_bound(currentLumi_);
    if ((iter != lumiSeen_.end()) && iter->first != currentLumi_) {

      auto elapsed = high_resolution_clock::now() - lastLumiLoad_;
      auto elapsed_ms = duration_cast<milliseconds>(elapsed).count();

      if (elapsed_ms >= nextLumiTimeoutMillis_) {
        std::string msg("Timeout reached, skipping lumisection(s) ");
        msg += std::to_string(currentLumi_) + " .. " +
               std::to_string(iter->first - 1);
        msg += ", currentLumi_ is now " + std::to_string(iter->first);

        logFileAction(msg);

        currentLumi_ = iter->first;
      }
    }
  }

  if (state_ == State::EOR_CLOSING) {
    // check if we parsed all lumis
    // n_lumi is both last lumi and the number of lumi
    // since lumis are indexed from 1

    // after all lumi have been pop()'ed
    // current lumi will become larger than the last lumi
    if (currentLumi_ > eor_.n_lumi) {
      state_ = State::EOR;
    }
  }

  if (state_ != old_state) {
    logFileAction("Streamer state changed: ",
                  std::to_string(old_state) + "->" + std::to_string(state_));
  }
}

void DQMFileIterator::logFileAction(const std::string& msg,
                                    const std::string& fileName) const {
  edm::LogAbsolute("fileAction") << std::setprecision(0) << edm::TimeOfDay()
                                 << "  " << msg << fileName;
  edm::FlushMessageLog();
}

void DQMFileIterator::updateWatchdog() {
  const char* x = getenv("WATCHDOG_FD");
  if (x) {
    int fd = atoi(x);
    write(fd, ".\n", 2);
  }
}

void DQMFileIterator::delay() {
  //logFileAction("Streamer waiting for the next LS.");

  updateWatchdog();
  usleep(delayMillis_ * 1000);
  updateWatchdog();
}

void DQMFileIterator::fillDescription(edm::ParameterSetDescription& desc) {

  desc.addUntracked<unsigned int>("runNumber")
      ->setComment("Run number passed via configuration file.");

  desc.addUntracked<unsigned int>("datafnPosition", 3)
      ->setComment("Data filename position in the positional arguments array 'data' in json file.");

  desc.addUntracked<std::string>("streamLabel")
      ->setComment("Stream label used in json discovery.");

  desc.addUntracked<uint32_t>("delayMillis")
      ->setComment("Number of milliseconds to wait between file checks.");

  desc.addUntracked<int32_t>("nextLumiTimeoutMillis", -1)->setComment(
      "Number of milliseconds to wait before switching to the next lumi "
      "section if the current is missing, -1 to disable.");

  desc.addUntracked<std::string>("runInputDir")
      ->setComment("Directory where the DQM files will appear.");
}

} /* end of namespace */
