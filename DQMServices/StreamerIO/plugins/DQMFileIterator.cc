#include "DQMFileIterator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

DQMFileIterator::DQMFileIterator() : state_(EOR) {}
DQMFileIterator::~DQMFileIterator() {}

void DQMFileIterator::initialise(int run, const std::string& path, const std::string& streamLabel) {
  run_ = run;
  streamLabel_ = streamLabel;
  run_path_ = str(boost::format("%s/run%06d") % path % run_);

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
  return str(boost::format("%s/run%06d_ls%04d%s.jsn") % run_path_ % run_ % lumi % streamLabel_);
}

std::string DQMFileIterator::make_path_eor() {
  return str(boost::format("%s/run%06d_ls0000_EoR.jsn") % run_path_ % run_);
}

std::string DQMFileIterator::make_path_data(const LumiEntry& lumi) {
  if (boost::starts_with(lumi.datafilename, "/")) return lumi.datafilename;

  boost::filesystem::path p(run_path_);
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
    edm::LogAbsolute("DQMStreamerReader") << "Checking eor file: " << fn_eor;

    if (boost::filesystem::exists(fn_eor)) {
      eor_ = EorEntry::load_json(fn_eor);

      edm::LogAbsolute("DQMStreamerReader") << "Loaded eor file: " << fn_eor;
    }
  }

  int nextLumi = lastLumiSeen_;  // initiate lumi
  for (;;) {
    nextLumi += 1;

    std::string fn = make_path_jsn(nextLumi);
    edm::LogAbsolute("DQMStreamerReader") << "Checking json file: " << fn;
    if (!boost::filesystem::exists(fn)) {
      // file not yet available
      break;
    }

    LumiEntry lumi = LumiEntry::load_json(fn, nextLumi);
    lastLumiSeen_ = nextLumi;
    queue_.push(lumi);

    edm::LogAbsolute("DQMStreamerReader") << "Loaded json file: " << fn;
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
    edm::LogAbsolute("DQMStreamerReader")
        << "Streamer state changed: " << old_state << " -> " << state_;
  }
}

} /* end of namespace */
