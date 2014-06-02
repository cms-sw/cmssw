#include "DQMFileIterator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include <queue>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/range.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

namespace edm {


DQMFileIterator::LumiEntry DQMFileIterator::LumiEntry::load_json(
    const std::string& filename, int lumiNumber) {
  boost::property_tree::ptree pt;
  read_json(filename, pt);

  LumiEntry lumi;

  // We rely on n_events to be the first item on the array...
  lumi.n_events = std::next(pt.get_child("data").begin(), 1)->second
      .get_value<std::size_t>();
  lumi.datafilename = std::next(pt.get_child("data").begin(), 2)->second
      .get_value<std::string>();
  lumi.definition = pt.get<std::string>("definition");
  lumi.source = pt.get<std::string>("source");

  lumi.ls = lumiNumber;
  return lumi;
}

DQMFileIterator::EorEntry DQMFileIterator::EorEntry::load_json(
    const std::string& filename) {
  boost::property_tree::ptree pt;
  read_json(filename, pt);

  EorEntry eor;

  // We rely on n_events to be the first item on the array...
  eor.n_events = std::next(pt.get_child("data").begin(), 1)->second
      .get_value<std::size_t>();
  eor.n_lumi = std::next(pt.get_child("data").begin(), 2)->second
      .get_value<std::size_t>();
  eor.datafilename = std::next(pt.get_child("data").begin(), 2)->second
      .get_value<std::string>();
  eor.definition = pt.get<std::string>("definition");
  eor.source = pt.get<std::string>("source");
  eor.loaded = true;

  return eor;
}


DQMFileIterator::DQMFileIterator(ParameterSet const& pset): state_(EOR) {
  runNumber_ = pset.getUntrackedParameter<unsigned int>("runNumber");
  runInputDir_ = pset.getUntrackedParameter<std::string>("runInputDir");
  streamLabel_ = pset.getUntrackedParameter<std::string>("streamLabel");
  delayMillis_ = pset.getUntrackedParameter<unsigned int>("delayMillis");

  reset();
}

DQMFileIterator::~DQMFileIterator() {}

void DQMFileIterator::reset() {
  runPath_ = str(boost::format("%s/run%06d") % runInputDir_ % runNumber_);

  eor_.loaded = false;
  state_ = State::OPEN;
  lastLumiSeen_ = 0;

  while (!queue_.empty()) {
    queue_.pop();
  }

  update_state();
}

DQMFileIterator::State DQMFileIterator::state() { return state_; }

const DQMFileIterator::LumiEntry& DQMFileIterator::front() {
  return queue_.front();
}

void DQMFileIterator::pop() { return queue_.pop(); }

bool DQMFileIterator::hasNext() {
  update_state();
  return !queue_.empty();
}

std::string DQMFileIterator::make_path_jsn(int lumi) {
  return str(boost::format("%s/run%06d_ls%04d%s.jsn") % runPath_ % runNumber_ % lumi % streamLabel_);
}

std::string DQMFileIterator::make_path_eor() {
  return str(boost::format("%s/run%06d_ls0000_EoR.jsn") % runPath_ % runNumber_);
}

std::string DQMFileIterator::make_path_data(const LumiEntry& lumi) {
  if (boost::starts_with(lumi.datafilename, "/")) return lumi.datafilename;

  boost::filesystem::path p(runPath_);
  p /= lumi.datafilename;
  return p.string();
}

void DQMFileIterator::collect() {
  // search filesystem to find available lumi section files
  // or the end of run file

  auto now = std::chrono::high_resolution_clock::now();
  auto last_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - last_collect_).count();

  if (last_ms < 100) return;

  last_collect_ = now;

  if (!eor_.loaded) {
    // end of run is not yet read
    std::string fn_eor = make_path_eor();
    logFileAction("Checking eor file: ", fn_eor);

    if (boost::filesystem::exists(fn_eor)) {
      eor_ = EorEntry::load_json(fn_eor);

      logFileAction("Loaded eor file: ", fn_eor);
    }
  }

  int nextLumi = lastLumiSeen_;  // initiate lumi
  for (;;) {
    nextLumi += 1;

    std::string fn = make_path_jsn(nextLumi);
    logFileAction("Checking json file: ", fn);

    if (!boost::filesystem::exists(fn)) {
      // file not yet available
      break;
    }

    LumiEntry lumi = LumiEntry::load_json(fn, nextLumi);
    lastLumiSeen_ = nextLumi;
    queue_.push(lumi);

    logFileAction("Loaded json file: ", fn);
  }
}

void DQMFileIterator::update_state() {
  collect();

  // now update the state
  State old_state = state_;

  if ((state_ == State::OPEN) && (eor_.loaded)) {
    state_ = State::EOR_CLOSING;
  }

  if (state_ == State::EOR_CLOSING) {
    if (int(eor_.n_lumi) <= lastLumiSeen_) {
      // last lumi number is also the number of lumis
      // ie lumi start from 1
      state_ = State::EOR;
    }
  }

  if (state_ != old_state) {
    logFileAction("Streamer state changed: ", std::to_string(old_state) + "->" + std::to_string(state_));
  }
}

void DQMFileIterator::logFileAction(const std::string& msg, const std::string& fileName) const {
  edm::LogAbsolute("fileAction") << std::setprecision(0) << edm::TimeOfDay() << "  " << msg << fileName;
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
  logFileAction("Streamer waiting for the next LS.");

  updateWatchdog();
  usleep(delayMillis_ * 1000);
  updateWatchdog();
}


void DQMFileIterator::fillDescription(
    ParameterSetDescription& desc) {

  desc.addUntracked<unsigned int>("runNumber")
      ->setComment("Run number passed via configuration file.");

  desc.addUntracked<std::string>("streamLabel")
      ->setComment("Stream label used in json discovery.");

  desc.addUntracked<unsigned int>("delayMillis")
      ->setComment("Number of milliseconds to wait between file checks.");

  desc.addUntracked<std::string>("runInputDir")
      ->setComment("Directory where the DQM files will appear.");
}


} /* end of namespace */
