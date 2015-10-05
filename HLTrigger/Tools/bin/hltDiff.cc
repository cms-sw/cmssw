/*      hltDiff: compare TriggerResults event by event
 *
 *      Compare two TriggerResults collections event by event.
 *      These can come from two collections in the same file(s), or from two different sets of files.
 */

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <memory>

#include <cstring>
#include <unistd.h>
#include <getopt.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"


void usage(std::ostream & out) {
  out << "\
usage: hltDiff -o|--old-files FILE1.ROOT [FILE2.ROOT ...] [-O|--old-process LABEL[:INSTANCE[:PROCESS]]]\n\
               -n|--new-files FILE1.ROOT [FILE2.ROOT ...] [-N|--new-process LABEL[:INSTANCE[:PROCESS]]]\n\
               [-m|--max-events MAXEVENTS] [-p|--prescales] [-v|--verbose] [-h|--help]\n\
\n\
  -o|--old-files FILE1.ROOT [FILE2.ROOT ...]\n\
      input file(s) with the old (reference) trigger results.\n\
\n\
  -O|--old-process PROCESS\n\
      process name of the collection with the old (reference) trigger results;\n\
      the default is to take the 'TriggerResults' from the last process.\n\
\n\
  -n|--new-files FILE1.ROOT [FILE2.ROOT ...]\n\
      input file(s) with the new trigger results to be compared with the reference;\n\
      to read these from a different collection in the same files as\n\
      the reference, use '-n -' and specify the collection with -N (see below).\n\
\n\
  -N|--new-process PROCESS\n\
      process name of the collection with the new (reference) trigger results;\n\
      the default is to take the 'TriggerResults' from the last process.\n\
\n\
  -m|--max-events MAXEVENTS\n\
      compare only the first MAXEVENTS events;\n\
      the default is to compare all the events in the original (reference) files.\n\
\n\
  -p|--prescales\n\
      do not ignore differences caused by HLTPrescaler modules.\n\
\n\
  -v|--verbose\n\
      be (more) verbose:\n\
      use once to print event-by-event comparison results;\n\
      use twice to print the trigger candidates of the affected filters;\n\
      use three times to print all the trigger candidates for the affected events.\n\
\n\
  -h|--help\n\
      print this help message, and exit." << std::endl;
}

void error(std::ostream & out) {
    out << "Try 'hltDiff --help' for more information." << std::endl;
}

void error(std::ostream & out, const char * message) {
  out << message << std::endl;
  error(out);
}

void error(std::ostream & out, const std::string & message) {
  out << message << std::endl;
  error(out);
}


class HLTConfigInterface {
public:
  virtual std::string const & processName() const = 0;
  virtual unsigned int size() const = 0;
  virtual unsigned int size(unsigned int trigger) const = 0;
  virtual std::string const & triggerName(unsigned int trigger) const = 0;
  virtual unsigned int triggerIndex(unsigned int trigger) const = 0;
  virtual std::string const & moduleLabel(unsigned int trigger, unsigned int module) const = 0;
  virtual std::string const & moduleType(unsigned int trigger, unsigned int module) const = 0;
  virtual bool prescaler(unsigned int trigger, unsigned int module) const = 0;
};


class HLTConfigDataEx : public HLTConfigInterface {
public:
  explicit HLTConfigDataEx(HLTConfigData data) :
    data_(data),
    moduleTypes_(size()),
    prescalers_(size())
  {
    for (unsigned int t = 0; t < data_.size(); ++t) {
      prescalers_[t].resize(size(t));
      moduleTypes_[t].resize(size(t));
      for (unsigned int m = 0; m < data_.size(t); ++m) {
        std::string type = data_.moduleType(moduleLabel(t, m));
        prescalers_[t][m] = (type == "HLTPrescaler");
        moduleTypes_[t][m] = &* moduleTypeSet_.insert(std::move(type)).first;
      }
    }
  }

  virtual std::string const & processName() const override {
    return data_.processName();
  }

  virtual unsigned int size() const override {
    return data_.size();
  }

  virtual unsigned int size(unsigned int trigger) const override {
    return data_.size(trigger);
  }

  virtual std::vector<std::string> const & triggerNames() const {
    return data_.triggerNames();
  }

  virtual std::string const & triggerName(unsigned int trigger) const override {
    return data_.triggerName(trigger);
  }

  virtual unsigned int triggerIndex(unsigned int trigger) const override {
    return trigger;
  }

  virtual std::string const & moduleLabel(unsigned int trigger, unsigned int module) const override {
    return data_.moduleLabel(trigger, module);
  }

  virtual std::string const & moduleType(unsigned int trigger, unsigned int module) const override {
    return * moduleTypes_.at(trigger).at(module);
  }

  virtual bool prescaler(unsigned int trigger, unsigned int module) const override {
    return prescalers_.at(trigger).at(module);
  }

private:
  HLTConfigData                                 data_;
  std::set<std::string>                         moduleTypeSet_;
  std::vector<std::vector<std::string const*>>  moduleTypes_;
  std::vector<std::vector<bool>>                prescalers_;
};


const char * event_state(bool state) {
  return state ? "accepted" : "rejected";
}


class HLTCommonConfig {
public:
  enum class Index {
    First = 0,
    Second = 1
  };

  class View : public HLTConfigInterface {
  public:
    View(HLTCommonConfig const & config, HLTCommonConfig::Index index) :
      config_(config),
      index_(index)
    { }

    virtual std::string const & processName() const override;
    virtual unsigned int size() const override;
    virtual unsigned int size(unsigned int trigger) const override;
    virtual std::string const & triggerName(unsigned int trigger) const override;
    virtual unsigned int triggerIndex(unsigned int trigger) const override;
    virtual std::string const & moduleLabel(unsigned int trigger, unsigned int module) const override;
    virtual std::string const & moduleType(unsigned int trigger, unsigned int module) const override;
    virtual bool prescaler(unsigned int trigger, unsigned int module) const override;

  private:
    HLTCommonConfig const & config_;
    Index index_;
  };


  HLTCommonConfig(HLTConfigDataEx const & first, HLTConfigDataEx const & second) :
    first_(first),
    second_(second),
    firstView_(*this, Index::First),
    secondView_(*this, Index::Second)
  {
    for (unsigned int f = 0; f < first.size(); ++f)
      for (unsigned int s = 0; s < second.size(); ++s)
        if (first.triggerName(f) == second.triggerName(s)) {
          triggerIndices_.push_back(std::make_pair(f, s));
          break;
        }
  }

  View const & getView(Index index) const {
    if (index == Index::First)
      return firstView_;
    else
      return secondView_;
  }

  std::string const & processName(Index index) const {
    if (index == Index::First)
      return first_.processName();
    else
      return second_.processName();
  }

  unsigned int size(Index index) const {
    return triggerIndices_.size();
  }

  unsigned int size(Index index, unsigned int trigger) const {
    if (index == Index::First)
      return first_.size(trigger);
    else
      return second_.size(trigger);
  }

  std::string const & triggerName(Index index, unsigned int trigger) const {
    if (index == Index::First)
      return first_.triggerName(triggerIndices_.at(trigger).first);
    else
      return second_.triggerName(triggerIndices_.at(trigger).second);
  }

  unsigned int triggerIndex(Index index, unsigned int trigger) const {
    if (index == Index::First)
      return triggerIndices_.at(trigger).first;
    else
      return triggerIndices_.at(trigger).second;
  }

  std::string const & moduleLabel(Index index, unsigned int trigger, unsigned int module) const {
    if (index == Index::First)
      return first_.moduleLabel(triggerIndices_.at(trigger).first, module);
    else
      return second_.moduleLabel(triggerIndices_.at(trigger).second, module);
  }

  std::string const & moduleType(Index index, unsigned int trigger, unsigned int module) const {
    if (index == Index::First)
      return first_.moduleType(triggerIndices_.at(trigger).first, module);
    else
      return second_.moduleType(triggerIndices_.at(trigger).second, module);
  }

  bool prescaler(Index index, unsigned int trigger, unsigned int module) const {
    if (index == Index::First)
      return first_.prescaler(triggerIndices_.at(trigger).first, module);
    else
      return second_.prescaler(triggerIndices_.at(trigger).second, module);
  }

private:
  HLTConfigDataEx const & first_;
  HLTConfigDataEx const & second_;

  View firstView_;
  View secondView_;

  std::vector<std::pair<unsigned int, unsigned int>> triggerIndices_;
};


std::string const & HLTCommonConfig::View::processName() const {
  return config_.processName(index_);
}

unsigned int HLTCommonConfig::View::size() const {
  return config_.size(index_);
}

unsigned int HLTCommonConfig::View::size(unsigned int trigger) const {
  return config_.size(index_, trigger);
}

std::string const & HLTCommonConfig::View::triggerName(unsigned int trigger) const {
  return config_.triggerName(index_, trigger);
}

unsigned int HLTCommonConfig::View::triggerIndex(unsigned int trigger) const {
  return config_.triggerIndex(index_, trigger);
}

std::string const & HLTCommonConfig::View::moduleLabel(unsigned int trigger, unsigned int module) const {
  return config_.moduleLabel(index_, trigger, module);
}

std::string const & HLTCommonConfig::View::moduleType(unsigned int trigger, unsigned int module) const {
  return config_.moduleType(index_, trigger, module);
}

bool HLTCommonConfig::View::prescaler(unsigned int trigger, unsigned int module) const {
  return config_.prescaler(index_, trigger, module);
}


enum State {
  Ready     = edm::hlt::Ready,
  Pass      = edm::hlt::Pass,
  Fail      = edm::hlt::Fail,
  Exception = edm::hlt::Exception,
  Prescaled,
  Invalid
};

const char * path_state(State state) {
  static const char * message[] = { "not run", "accepted", "rejected", "exception", "prescaled", "invalid" };

  if (state > 0 and state < Invalid)
    return message[state];
  else
    return message[Invalid];
}

inline
State prescaled_state(int state, int path, int module, HLTConfigInterface const & config) {
  if (state == Fail and config.prescaler(path, module))
    return Prescaled;
  return (State) state;
}

void print_detailed_path_state(std::ostream & out, State state, int path, int module, HLTConfigInterface const & config) {
  auto const & label = config.moduleLabel(path, module);
  auto const & type  = config.moduleType(path, module);

  out << "'" << path_state(state) << "'";
  if (state == Fail)
    out << " by module " << module << " '" << label << "' [" << type << "]";
  else if (state == Exception)
    out << " at module " << module << " '" << label << "' [" << type << "]";
}

void print_trigger_candidates(std::ostream & out, trigger::TriggerEvent const & summary, edm::InputTag const & filter) {
  // find the index of the collection of trigger candidates corresponding to the filter
  unsigned int index = summary.filterIndex(filter);
  
  if (index >= summary.sizeFilters()) {
    // the collection of trigger candidates corresponding to the filter could not be found
    out << "            not found\n";
    return;
  }

  if (summary.filterKeys(index).empty()) {
    // the collection of trigger candidates corresponding to the filter is empty
    out << "            none\n";
    return;
  }

  for (unsigned int i = 0; i < summary.filterKeys(index).size(); ++i) {
    auto key = summary.filterKeys(index)[i];
    auto id  = summary.filterIds(index)[i];
    trigger::TriggerObject const & candidate = summary.getObjects().at(key);
    out << "            " 
        << "filter id: " << id               << ", "
        << "object id: " << candidate.id()   << ", "
        << "pT: "        << candidate.pt()   << ", "
        << "eta: "       << candidate.eta()  << ", "
        << "phi: "       << candidate.phi()  << ", "
        << "mass: "      << candidate.mass() << "\n";
  }
}

void print_trigger_collection(std::ostream & out, trigger::TriggerEvent const & summary, std::string const & tag) {
  auto iterator = std::find(summary.collectionTags().begin(), summary.collectionTags().end(), tag);
  if (iterator == summary.collectionTags().end()) {
    // the collection of trigger candidates could not be found
    out << "            not found\n";
    return;
  }

  unsigned int index = iterator - summary.collectionTags().begin();
  unsigned int begin = (index == 0) ? 0 : summary.collectionKey(index - 1);
  unsigned int end   = summary.collectionKey(index);

  if (end == begin) {
    // the collection of trigger candidates is empty
    out << "            none\n";
    return;
  }

  for (unsigned int key = begin; key < end; ++key) {
    trigger::TriggerObject const & candidate = summary.getObjects().at(key);
    out << "            " 
        << "object id: " << candidate.id()   << ", "
        << "pT: "        << candidate.pt()   << ", "
        << "eta: "       << candidate.eta()  << ", "
        << "phi: "       << candidate.phi()  << ", "
        << "mass: "      << candidate.mass() << "\n";
  }
}


std::string getProcessNameFromBranch(std::string const & branch) {
  std::vector<boost::iterator_range<std::string::const_iterator>> tokens;
  boost::split(tokens, branch, boost::is_any_of("_."), boost::token_compress_off);
  return boost::copy_range<std::string>(tokens[3]);
}

std::unique_ptr<HLTConfigDataEx> getHLTConfigData(fwlite::EventBase const & event, std::string process) {
  auto const & history = event.processHistory();
  if (process.empty()) {
    // determine the process name from the most recent "TriggerResults" object
    auto const & branch  = event.getBranchNameFor( edm::Wrapper<edm::TriggerResults>::typeInfo(), "TriggerResults", "", process.c_str() );
    process = getProcessNameFromBranch( branch );
  }

  edm::ProcessConfiguration config;
  if (not history.getConfigurationForProcess(process, config)) {
    std::cerr << "error: the process " << process << " is not in the Process History" << std::endl;
    exit(1);
  }
  const edm::ParameterSet* pset = edm::pset::Registry::instance()->getMapped(config.parameterSetID());
  if (pset == nullptr) {
    std::cerr << "error: the configuration for the process " << process << " is not available in the Provenance" << std::endl;
    exit(1);
  }
  return std::unique_ptr<HLTConfigDataEx>(new HLTConfigDataEx(HLTConfigData(pset)));
}


struct TriggerDiff {
  TriggerDiff() : count(0), gained(0), lost(0), internal(0) { }

  unsigned int count;
  unsigned int gained;
  unsigned int lost;
  unsigned int internal;

  static
  std::string format(unsigned int value, char sign = '+') {
    if (value == 0)
      return std::string("-");

    char buffer[12];        // sign, 10 digits, null
    memset(buffer, 0, 12);

    unsigned int digit = 10;
    while (value > 0) {
      buffer[digit] = value % 10 + 48;
      value /= 10;
      --digit;
    }
    buffer[digit] = sign;

    return std::string(buffer + digit);
  }
};

std::ostream & operator<<(std::ostream & out, TriggerDiff diff) {
  out << std::setw(12) << diff.count
      << std::setw(12) << TriggerDiff::format(diff.gained, '+')
      << std::setw(12) << TriggerDiff::format(diff.lost, '-')
      << std::setw(12) << TriggerDiff::format(diff.internal, '~');
  return out;
}


bool check_file(std::string const & file) {
  boost::filesystem::path p(file);

  // check if the file exists
  if (not boost::filesystem::exists(p))
    return false;

  // resolve the file name to canonical form
  p = boost::filesystem::canonical(p);
  if (not boost::filesystem::exists(p))
    return false;

  // check for a regular file
  if (not boost::filesystem::is_regular_file(p))
    return false;

  return true;
}


bool check_files(std::vector<std::string> const & files) {
  bool flag = true;
  for (auto const & file: files)
    if (not check_file(file)) {
      flag = false;
      std::cerr << "hltDiff: error: file " << file << " does not exist, or is not a regular file." << std::endl;
    }
  return flag;
}


void compare(std::vector<std::string> const & old_files, std::string const & old_process,
             std::vector<std::string> const & new_files, std::string const & new_process,
             unsigned int max_events, bool ignore_prescales, int verbose) {

  std::shared_ptr<fwlite::ChainEvent> old_events;
  std::shared_ptr<fwlite::ChainEvent> new_events;

  if (check_files(old_files))
    old_events = std::make_shared<fwlite::ChainEvent>(old_files);
  else
    return;

  if (new_files.size() == 1 and new_files[0] == "-")
    new_events = old_events;
  else if (check_files(new_files))
    new_events = std::make_shared<fwlite::ChainEvent>(new_files);
  else
    return;

  std::unique_ptr<HLTConfigDataEx> old_config_data;
  std::unique_ptr<HLTConfigDataEx> new_config_data;
  std::unique_ptr<HLTCommonConfig> common_config;
  HLTConfigInterface const * old_config = nullptr;
  HLTConfigInterface const * new_config = nullptr;

  unsigned int counter = 0;
  unsigned int affected = 0;
  bool new_run = true;
  std::vector<TriggerDiff> differences;

  // loop over the reference events
  for (old_events->toBegin(); not old_events->atEnd(); ++(*old_events)) {

    // seek the same event in the "new" files
    edm::EventID const& id = old_events->id();
    if (new_events != old_events and not new_events->to(id)) {
      std::cerr << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event() << ": not found in the 'new' files, skipping." << std::endl;
      continue;
    }

    // read the TriggerResults and TriggerEvent
    fwlite::Handle<edm::TriggerResults> old_results_h;
    old_results_h.getByLabel<fwlite::Event>(* old_events->event(), "TriggerResults", "", old_process.c_str());
    auto const & old_results = * old_results_h;

    fwlite::Handle<trigger::TriggerEvent> old_summary_h;
    old_summary_h.getByLabel<fwlite::Event>(* old_events->event(), "hltTriggerSummaryAOD", "", old_process.c_str());
    auto const & old_summary = * old_summary_h;

    fwlite::Handle<edm::TriggerResults> new_handle;
    new_handle.getByLabel<fwlite::Event>(* new_events->event(), "TriggerResults", "", new_process.c_str());
    auto const & new_results = * new_handle;

    fwlite::Handle<trigger::TriggerEvent> new_summary_h;
    new_summary_h.getByLabel<fwlite::Event>(* new_events->event(), "hltTriggerSummaryAOD", "", new_process.c_str());
    auto const & new_summary = * new_summary_h;

    // initialise the trigger configuration
    if (new_run) {
      new_run = false;
      old_events->fillParameterSetRegistry();
      new_events->fillParameterSetRegistry();

      old_config_data = getHLTConfigData(* old_events->event(), old_process);
      new_config_data = getHLTConfigData(* new_events->event(), new_process);
      if (new_config_data->triggerNames() == old_config_data->triggerNames()) {
        old_config = old_config_data.get();
        new_config = new_config_data.get();
      } else {
        common_config = std::unique_ptr<HLTCommonConfig>(new HLTCommonConfig(*old_config_data, *new_config_data));
        old_config = & common_config->getView(HLTCommonConfig::Index::First);
        new_config = & common_config->getView(HLTCommonConfig::Index::Second);
        std::cerr << "Warning: old and new TriggerResults come from different HLT menus. Only the common triggers will be compared:" << std::endl;
        for (unsigned int i = 0; i < old_config->size(); ++i)
          std::cerr << "    " << old_config->triggerName(i) << std::endl;
        std::cerr << std::endl;
      }

      differences.clear();
      differences.resize(old_config->size());
    }

    // compare the TriggerResults
    bool needs_header = true;
    bool affected_event = false;
    for (unsigned int p = 0; p < old_config->size(); ++p) {
      // FIXME explicitly converting the indices is a hack, it should be properly encapsulated instead
      unsigned int old_index = old_config->triggerIndex(p);
      unsigned int new_index = new_config->triggerIndex(p);
      State old_state = prescaled_state(old_results.state(old_index), p, old_results.index(old_index), * old_config);
      State new_state = prescaled_state(new_results.state(new_index), p, new_results.index(new_index), * new_config);

      if (old_state == Pass)
        ++differences[p].count;

      bool flag = false;
      if (not ignore_prescales or (old_state != Prescaled and new_state != Prescaled)) {
        if (old_state == Pass and new_state != Pass) {
          ++differences[p].lost;
          flag = true;
        } else if (old_state != Pass and new_state == Pass) {
          ++differences[p].gained;
          flag = true;
        } else if (old_results.index(old_index) != new_results.index(new_index)) {
          ++differences[p].internal;
          flag = true;
        }
      }

      if (flag) {
        affected_event = true;

        if (verbose > 0) {
          if (needs_header) {
            needs_header = false;
            std::cout << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event() << ": "
                      << "old result is '" << event_state(old_results.accept()) << "', "
                      << "new result is '" << event_state(new_results.accept()) << "'"
                      << std::endl;
          }
          // print the Trigger path and filter responsible for the discrepancy
          std::cout << "    Path " << old_config->triggerName(p) << ":\n"
                    << "        old state is ";
          print_detailed_path_state(std::cout, old_state, p, old_results.index(old_index), * old_config);
          std::cout << ",\n"
                    << "        new state is ";
          print_detailed_path_state(std::cout, new_state, p, new_results.index(new_index), * new_config);
          std::cout << std::endl;
        }
        if (verbose > 1) {
          // print TriggerObjects for the filter responsible for the discrepancy
          unsigned int module = std::min(old_results.index(old_index), new_results.index(new_index));
          std::cout << "    Filter " << old_config->moduleLabel(p, module) << ":\n";
          std::cout << "        old trigger candidates:\n";
          print_trigger_candidates(std::cout, old_summary, edm::InputTag(old_config->moduleLabel(p, module), "", old_config->processName()));
          std::cout << "        new trigger candidates:\n";
          print_trigger_candidates(std::cout, new_summary, edm::InputTag(new_config->moduleLabel(p, module), "", new_config->processName()));
        }
        if (verbose > 0)
          std::cout << std::endl;
      }
    }
    if (affected_event)
      ++affected;

    // compare the TriggerEvent
    if (affected_event and verbose > 2) {
      std::set<std::string> names;
      names.insert(old_summary.collectionTags().begin(), old_summary.collectionTags().end());
      names.insert(new_summary.collectionTags().begin(), new_summary.collectionTags().end());
      for (auto const & collection: names) {
        std::cout << "    Collection " << collection << ":\n";
        std::cout << "        old trigger candidates:\n";
        print_trigger_collection(std::cout, old_summary, collection);
        std::cout << "        new trigger candidates:\n";
        print_trigger_collection(std::cout, new_summary, collection);
        std::cout << std::endl;
      }
    }

    ++counter;
    if (max_events and counter >= max_events)
      break;
  }

  if (not counter) {
    std::cout << "There are no common events between the old and new files." << std::endl;
  } else {
    std::cout << "Found " << affected << " events out of " << counter << " with differences:\n" << std::endl;
    std::cout << std::setw(12) << "Events" << std::setw(12) << "Accepted" << std::setw(12) << "Gained" << std::setw(12) << "Lost" << std::setw(12) << "Other" << "  " << "Trigger" << std::endl;
    for (unsigned int p = 0; p < old_config->size(); ++p)
      std::cout << std::setw(12) << counter << differences[p] << "  " << old_config->triggerName(p) << std::endl;
  }
}


int main(int argc, char ** argv) {
  // options
  const char optstring[] = "o:O:n:N:m:pvh";
  const option longopts[] = {
    option{ "old-files",    required_argument,  nullptr, 'o' },
    option{ "old-process",  required_argument,  nullptr, 'O' },
    option{ "new-files",    required_argument,  nullptr, 'n' },
    option{ "new-process",  required_argument,  nullptr, 'N' },
    option{ "max-events",   required_argument,  nullptr, 'm' },
    option{ "prescales",    no_argument,        nullptr, 'p' },
    option{ "verbose",      no_argument,        nullptr, 'v' },
    option{ "help",         no_argument,        nullptr, 'h' },
  };

  // default values
  std::vector<std::string>  old_files;
  std::string               old_process("");
  std::vector<std::string>  new_files;
  std::string               new_process("");
  unsigned int              max_events = 0;
  bool                      ignore_prescales = true;
  unsigned int              verbose = 0;

  // parse the command line options
  int c = -1;
  while ((c = getopt_long(argc, argv, optstring, longopts, nullptr)) != -1) {
    switch (c) {
      case 'o':
        old_files.emplace_back(optarg);
        while (optind < argc) {
          if (argv[optind][0] == '-')
            break;
          old_files.emplace_back(argv[optind]);
          ++optind;
        }
        break;

      case 'O':
        old_process = optarg;
        break;

      case 'n':
        new_files.emplace_back(optarg);
        while (optind < argc) {
          if (argv[optind][0] == '-')
            break;
          new_files.emplace_back(argv[optind]);
          ++optind;
        }
        break;

      case 'N':
        new_process = optarg;
        break;

      case 'm':
        max_events = atoi(optarg);
        break;

      case 'p':
        ignore_prescales = false;
        break;

      case 'v':
        ++verbose;
        break;

      case 'h':
        usage(std::cerr);
        exit(0);
        break;

      default:
        error(std::cerr);
        exit(1);
        break;
    }
  }

  if (old_files.empty()) {
    error(std::cerr, "hltDiff: please specify the 'old' file(s)");
    exit(1);
  }
  if (new_files.empty()) {
    error(std::cerr, "hltDiff: please specify the 'new' file(s)");
    exit(1);
  }

  compare(old_files, old_process, new_files, new_process, max_events, ignore_prescales, verbose);

  return 0;
}
