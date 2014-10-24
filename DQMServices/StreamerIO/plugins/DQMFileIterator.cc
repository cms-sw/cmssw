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

  lumi.file_ls = lumiNumber;
  lumi.datafilename = std::next(pt.get_child("data").begin(), datafn_position)
    ->second.get_value<std::string>();

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

  forceFileCheckTimeoutMillis_ = 5015;
  reset();

  if (mon_.isAvailable()) {
    ptree doc;
    doc.put("run", runNumber_);
    mon_->outputUpdate(doc);
    updateMonitoring();
  }
}

DQMFileIterator::~DQMFileIterator() {}

void DQMFileIterator::reset() {
  runPath_ = str(boost::format("%s/run%06d") % runInputDir_ % runNumber_);

  eor_.loaded = false;
  state_ = State::OPEN;
  nextLumiNumber_ = 1;
  lumiSeen_.clear();

  lastLumiLoad_ = std::chrono::high_resolution_clock::now();

  collect(true);
  update_state();
}

DQMFileIterator::State DQMFileIterator::state() { return state_; }

const DQMFileIterator::LumiEntry DQMFileIterator::open() {
  LumiEntry& lumi = lumiSeen_[nextLumiNumber_];
  advanceToLumi(nextLumiNumber_ + 1);

  lumi.state = "open: file iterator";
  return lumi;
}

bool DQMFileIterator::lumiReady() {
  if (lumiSeen_.find(nextLumiNumber_) != lumiSeen_.end()) {
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

  unsigned int currentLumi = nextLumiNumber_;

  nextLumiNumber_ = lumi;
  lastLumiLoad_ = std::chrono::high_resolution_clock::now();

  if (mon_.isAvailable()) {
    // report the successful lumi file open
    ptree doc;
    doc.put("lumi", currentLumi);
    mon_->outputUpdate(doc);
    updateMonitoring();
  }
}

void DQMFileIterator::updateMonitoring() {
  if (! mon_.isAvailable())
    return;

  ptree children;
  auto iter = lumiSeen_.begin();
  for (; iter != lumiSeen_.end(); ++iter) {
    ptree lumi;
    lumi.put("filename", iter->second.filename);
    lumi.put("file_ls", iter->second.file_ls);
    lumi.put("state", iter->second.state);

    children.add_child(std::to_string(iter->first), lumi);
  }

  ptree doc;
  doc.add_child("lumiSeen", children);
  mon_->outputUpdate(doc);
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
  if ((!ignoreTimers) && (last_ms >= 0) && (last_ms < 100)) {
    return;
  }

  // check if directory changed
  std::time_t mtime_now = boost::filesystem::last_write_time(runPath_);

  if ((!ignoreTimers) && (last_ms < forceFileCheckTimeoutMillis_) && (mtime_now  == runPathMTime_)) {
    //logFileAction("Directory hasn't changed.");
    return;
  } else {
    //logFileAction("Directory changed, updating.");
  }

  runPathMTime_ = mtime_now;
  runPathLastCollect_ = now;

  using boost::filesystem::directory_iterator;
  using boost::filesystem::directory_entry;

  std::string fn_eor;

  directory_iterator dend;
  for (directory_iterator di(runPath_); di != dend; ++di) {
    const boost::regex fn_re("run(\\d+)_ls(\\d+)_([a-zA-Z0-9]+)(_.*)?\\.jsn");

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
      if ((lumi == 0) && (label == "EoR") && (!eor_.loaded)) {
        fn_eor = fn;
        continue;
      }

      // check if lumi is loaded
      if (lumiSeen_.find(lumi) != lumiSeen_.end()) {
        continue;  // already loaded
      }

      // check if this belongs to us
      if (label != streamLabel_) {
        std::string msg("Found and skipped json file (stream label mismatch, ");
        msg += label + " [files] != " + streamLabel_ + " [config]";
        msg += "): ";
        logFileAction(msg, fn);
        continue;
      }

      try {
        LumiEntry lumi_jsn = LumiEntry::load_json(fn, lumi, datafnPosition_);
        lumiSeen_.emplace(lumi, lumi_jsn);
        logFileAction("Found and loaded json file: ", fn);
      } catch (const std::exception& e) {
        // don't reset the mtime, keep it waiting
        std::string msg("Found, tried to load the json, but failed (");
        msg += e.what();
        msg += "): ";
        logFileAction(msg, fn);
      }
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
    auto iter = lumiSeen_.lower_bound(nextLumiNumber_);
    if ((iter != lumiSeen_.end()) && iter->first != nextLumiNumber_) {

      auto elapsed = high_resolution_clock::now() - lastLumiLoad_;
      auto elapsed_ms = duration_cast<milliseconds>(elapsed).count();

      if (elapsed_ms >= nextLumiTimeoutMillis_) {
        std::string msg("Timeout reached, skipping lumisection(s) ");
        msg += std::to_string(nextLumiNumber_) + " .. " +
               std::to_string(iter->first - 1);
        msg += ", nextLumiNumber_ is now " + std::to_string(iter->first);

        logFileAction(msg);

        nextLumiNumber_ = iter->first;
      }
    }
  }

  if (state_ == State::EOR_CLOSING) {
    // check if we parsed all lumis
    // n_lumi is both last lumi and the number of lumi
    // since lumis are indexed from 1

    // after all lumi have been pop()'ed
    // current lumi will become larger than the last lumi
    if (nextLumiNumber_ > eor_.n_lumi) {
      state_ = State::EOR;
    }
  }

  if (state_ != old_state) {
    logFileAction("Streamer state changed: ",
                  std::to_string(old_state) + "->" + std::to_string(state_));

    updateMonitoring();
  }
}

void DQMFileIterator::logFileAction(const std::string& msg,
                                    const std::string& fileName) const {
  edm::LogAbsolute("fileAction") << std::setprecision(0) << edm::TimeOfDay()
                                 << "  " << msg << fileName;
  edm::FlushMessageLog();
}

void DQMFileIterator::logLumiState(const LumiEntry& lumi, const std::string& msg) {
  if (lumiSeen_.find(lumi.file_ls) != lumiSeen_.end()) {
    lumiSeen_[lumi.file_ls].state = msg;
  } else {
    logFileAction("Internal error: referenced lumi is not the map.");
  }
}

void DQMFileIterator::delay() {
  //logFileAction("Streamer waiting for the next LS.");

  if (mon_.isAvailable())
    mon_->keepAlive();

  usleep(delayMillis_ * 1000);

  if (mon_.isAvailable())
    mon_->keepAlive();
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
