#include "DQMFileIterator.h"
#include "DQMMonitoringService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include <filesystem>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/regex.hpp>

#include <fmt/printf.h>

namespace dqmservices {

  DQMFileIterator::LumiEntry DQMFileIterator::LumiEntry::load_json(const std::string& run_path,
                                                                   const std::string& filename,
                                                                   int lumiNumber,
                                                                   int datafn_position) {
    boost::property_tree::ptree pt;
    read_json(filename, pt);

    LumiEntry lumi;
    lumi.filename = filename;
    lumi.run_path = run_path;

    lumi.n_events_processed = std::next(pt.get_child("data").begin(), 0)->second.get_value<std::size_t>();

    lumi.n_events_accepted = std::next(pt.get_child("data").begin(), 1)->second.get_value<std::size_t>();

    lumi.file_ls = lumiNumber;

    if (datafn_position >= 0) {
      lumi.datafn = std::next(pt.get_child("data").begin(), datafn_position)->second.get_value<std::string>();
    }

    return lumi;
  }

  std::string DQMFileIterator::LumiEntry::get_data_path() const {
    if (boost::starts_with(datafn, "/"))
      return datafn;

    std::filesystem::path p(run_path);
    p /= datafn;
    return p.string();
  }

  std::string DQMFileIterator::LumiEntry::get_json_path() const { return filename; }

  // Contents of Eor json file are ignored for the moment.
  // This function will not be called.
  DQMFileIterator::EorEntry DQMFileIterator::EorEntry::load_json(const std::string& run_path,
                                                                 const std::string& filename) {
    boost::property_tree::ptree pt;
    read_json(filename, pt);

    EorEntry eor;
    eor.filename = filename;
    eor.run_path = run_path;

    // We rely on n_events to be the first item on the array...
    eor.n_events = std::next(pt.get_child("data").begin(), 1)->second.get_value<std::size_t>();
    eor.n_lumi = std::next(pt.get_child("data").begin(), 2)->second.get_value<std::size_t>();

    eor.loaded = true;
    return eor;
  }

  DQMFileIterator::DQMFileIterator(edm::ParameterSet const& pset) : state_(EOR) {
    runNumber_ = pset.getUntrackedParameter<unsigned int>("runNumber");
    datafnPosition_ = pset.getUntrackedParameter<unsigned int>("datafnPosition");
    runInputDir_ = pset.getUntrackedParameter<std::string>("runInputDir");
    streamLabel_ = pset.getUntrackedParameter<std::string>("streamLabel");
    delayMillis_ = pset.getUntrackedParameter<uint32_t>("delayMillis");
    nextLumiTimeoutMillis_ = pset.getUntrackedParameter<int32_t>("nextLumiTimeoutMillis");

    // scan one mode
    flagScanOnce_ = pset.getUntrackedParameter<bool>("scanOnce");

    forceFileCheckTimeoutMillis_ = 5015;
    reset();
  }

  void DQMFileIterator::reset() {
    runPath_.clear();

    std::vector<std::string> tokens;
    boost::split(tokens, runInputDir_, boost::is_any_of(":"));

    for (const auto& token : tokens) {
      runPath_.push_back(fmt::sprintf("%s/run%06d", token, runNumber_));
    }

    eor_.loaded = false;
    state_ = State::OPEN;
    nextLumiNumber_ = 1;
    lumiSeen_.clear();
    filesSeen_.clear();

    lastLumiLoad_ = std::chrono::high_resolution_clock::now();

    collect(true);
    update_state();

    if (mon_.isAvailable()) {
      boost::property_tree::ptree doc;
      doc.put("run", runNumber_);
      doc.put("next_lumi", nextLumiNumber_);
      doc.put("fi_state", std::to_string(state_));
      mon_->outputUpdate(doc);
    }
  }

  DQMFileIterator::LumiEntry DQMFileIterator::open() {
    LumiEntry& lumi = lumiSeen_[nextLumiNumber_];
    advanceToLumi(nextLumiNumber_ + 1, "open: file iterator");
    return lumi;
  }

  bool DQMFileIterator::lumiReady() {
    if (lumiSeen_.find(nextLumiNumber_) != lumiSeen_.end()) {
      return true;
    }

    return false;
  }

  unsigned int DQMFileIterator::lastLumiFound() {
    if (!lumiSeen_.empty()) {
      return lumiSeen_.rbegin()->first;
    }

    return 1;
  }

  void DQMFileIterator::advanceToLumi(unsigned int lumi, std::string reason) {
    unsigned int currentLumi = nextLumiNumber_;

    nextLumiNumber_ = lumi;
    lastLumiLoad_ = std::chrono::high_resolution_clock::now();

    auto iter = lumiSeen_.lower_bound(currentLumi);

    while ((iter != lumiSeen_.end()) && ((iter->first) < nextLumiNumber_)) {
      iter->second.state = reason;
      monUpdateLumi(iter->second);

      ++iter;
    }

    if (mon_.isAvailable()) {
      // report the successful lumi file open
      boost::property_tree::ptree doc;
      doc.put("next_lumi", nextLumiNumber_);
      mon_->outputUpdate(doc);
    }
  }

  void DQMFileIterator::monUpdateLumi(const LumiEntry& lumi) {
    if (!mon_.isAvailable())
      return;

    boost::property_tree::ptree doc;
    doc.put(fmt::sprintf("extra.lumi_seen.lumi%06d", lumi.file_ls), lumi.state);
    mon_->outputUpdate(doc);
  }

  unsigned DQMFileIterator::mtimeHash() const {
    unsigned mtime_now = 0;

    for (const auto& path : runPath_) {
      if (!std::filesystem::exists(path))
        continue;

      auto write_time = std::filesystem::last_write_time(path);
      mtime_now =
          mtime_now ^ std::chrono::duration_cast<std::chrono::microseconds>(write_time.time_since_epoch()).count();
    }

    return mtime_now;
  }

  void DQMFileIterator::collect(bool ignoreTimers) {
    // search filesystem to find available lumi section files
    // or the end of run files

    auto now = std::chrono::high_resolution_clock::now();
    auto last_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - runPathLastCollect_).count();

    // don't refresh if it's too soon
    if ((!ignoreTimers) && (last_ms >= 0) && (last_ms < 100)) {
      return;
    }

    // check if directory changed
    auto mtime_now = mtimeHash();

    if ((!ignoreTimers) && (last_ms < forceFileCheckTimeoutMillis_) && (mtime_now == runPathMTime_)) {
      // logFileAction("Directory hasn't changed.");
      return;
    } else {
      // logFileAction("Directory changed, updating.");
    }

    runPathMTime_ = mtime_now;
    runPathLastCollect_ = now;

    using std::filesystem::directory_entry;
    using std::filesystem::directory_iterator;

    std::string fn_eor;

    for (const auto& runPath : runPath_) {
      if (!std::filesystem::exists(runPath)) {
        logFileAction("Directory does not exist: ", runPath);

        continue;
      }

      directory_iterator dend;
      for (directory_iterator di(runPath); di != dend; ++di) {
        const boost::regex fn_re("run(\\d+)_ls(\\d+)_([a-zA-Z0-9]+)(_.*)?\\.jsn");

        const std::string filename = di->path().filename().string();
        const std::string fn = di->path().string();

        if (filesSeen_.find(filename) != filesSeen_.end()) {
          continue;
        }

        boost::smatch result;
        if (boost::regex_match(filename, result, fn_re)) {
          unsigned int run = std::stoi(result[1]);
          unsigned int lumi = std::stoi(result[2]);
          std::string label = result[3];

          filesSeen_.insert(filename);

          if (run != runNumber_)
            continue;

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
            LumiEntry lumi_jsn = LumiEntry::load_json(runPath, fn, lumi, datafnPosition_);
            lumiSeen_.emplace(lumi, lumi_jsn);
            logFileAction("Found and loaded json file: ", fn);

            monUpdateLumi(lumi_jsn);
          } catch (const std::exception& e) {
            // don't reset the mtime, keep it waiting
            filesSeen_.erase(filename);

            std::string msg("Found, tried to load the json, but failed (");
            msg += e.what();
            msg += "): ";
            logFileAction(msg, fn);
          }
        }
      }
    }

    if ((!fn_eor.empty()) or flagScanOnce_) {
      if (!fn_eor.empty()) {
        logFileAction("EoR file found: ", fn_eor);
      }

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
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    State old_state = state_;

    // in scanOnce mode we don't do repeated scans
    // whatever found at reset() is be used
    if (!flagScanOnce_) {
      collect(false);
    }

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
          msg += std::to_string(nextLumiNumber_) + " .. " + std::to_string(iter->first - 1);
          msg += ", nextLumiNumber_ is now " + std::to_string(iter->first);
          logFileAction(msg);

          advanceToLumi(iter->first, "skipped: timeout");
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
      logFileAction("Streamer state changed: ", std::to_string(old_state) + "->" + std::to_string(state_));

      if (mon_) {
        boost::property_tree::ptree doc;
        doc.put("fi_state", std::to_string(state_));
        mon_->outputUpdate(doc);
      }
    }
  }

  void DQMFileIterator::logFileAction(const std::string& msg, const std::string& fileName) const {
    edm::LogAbsolute("fileAction") << std::setprecision(0) << edm::TimeOfDay() << "  " << msg << fileName;
    edm::FlushMessageLog();
  }

  void DQMFileIterator::logLumiState(const LumiEntry& lumi, const std::string& msg) {
    if (lumiSeen_.find(lumi.file_ls) != lumiSeen_.end()) {
      lumiSeen_[lumi.file_ls].state = msg;

      monUpdateLumi(lumiSeen_[lumi.file_ls]);
    } else {
      logFileAction("Internal error: referenced lumi is not the map.");
    }
  }

  void DQMFileIterator::delay() {
    if (mon_.isAvailable())
      mon_->keepAlive();

    usleep(delayMillis_ * 1000);
  }

  void DQMFileIterator::fillDescription(edm::ParameterSetDescription& desc) {
    desc.addUntracked<unsigned int>("runNumber")->setComment("Run number passed via configuration file.");

    desc.addUntracked<unsigned int>("datafnPosition", 3)
        ->setComment(
            "Data filename position in the positional arguments array 'data' in "
            "json file.");

    desc.addUntracked<std::string>("streamLabel")->setComment("Stream label used in json discovery.");

    desc.addUntracked<uint32_t>("delayMillis")->setComment("Number of milliseconds to wait between file checks.");

    desc.addUntracked<int32_t>("nextLumiTimeoutMillis", -1)
        ->setComment(
            "Number of milliseconds to wait before switching to the next lumi "
            "section if the current is missing, -1 to disable.");

    desc.addUntracked<bool>("scanOnce", false)
        ->setComment(
            "Don't repeat file scans: use what was found during the initial scan. "
            "EOR file is ignored and the state is set to 'past end of run'.");

    desc.addUntracked<std::string>("runInputDir")->setComment("Directory where the DQM files will appear.");
  }

}  // namespace dqmservices
