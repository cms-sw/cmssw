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

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"


void usage(std::ostream & out) {
  out << "\
usage: hltDiff -o|--old-files FILE1.ROOT [FILE2.ROOT ...] [-O|--old-label LABEL[:INSTANCE[:PROCESS]]]\n\
               -n|--new-files FILE1.ROOT [FILE2.ROOT ...] [-N|--new-label LABEL[:INSTANCE[:PROCESS]]]\n\
               [-m|--max-events MAXEVENTS] [-p|--prescales] [-v|--verbose] [-h|--help]\n\
\n\
  -o|--old-files FILE1.ROOT [FILE2.ROOT ...]\n\
      input file(s) with the old (reference) trigger results.\n\
\n\
  -O|--old-label LABEL[:INSTANCE[:PROCESS]]\n\
      collection with the old (reference) trigger results;\n\
      the default is 'TriggerResults' (without any instance or process name).\n\
\n\
  -n|--new-files FILE1.ROOT [FILE2.ROOT ...]\n\
      input file(s) with the new trigger results to be compared with the reference;\n\
      to read these from a different collection in the same files as\n\
      the reference, use '-n -' and specify the collection with -N (see below).\n\
\n\
  -N|--new-label LABEL[:INSTANCE[:PROCESS]]\n\
      collection with the new trigger results to be compared with the reference;\n\
      the default is 'TriggerResults' (without any instance or process name).\n\
\n\
  -m|--max-events MAXEVENTS\n\
      compare only the first MAXEVENTS events;\n\
      the default is to compare all the events in the original (reference) files.\n\
\n\
  -p|--prescales\n\
      do not ignore differences caused by HLTPrescaler modules.\n\
\n\
  -v|--verbose\n\
      be verbose: print event-by-event comparison results.\n\
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


class HLTConfigDataEx : public HLTConfigData {
public:
  explicit HLTConfigDataEx(HLTConfigData const & data) :
    HLTConfigData(data),
    moduleTypes_(size()),
    prescalers_(size())
  {
    for (unsigned int t = 0; t < size(); ++t) {
      prescalers_[t].resize(size(t));
      moduleTypes_[t].resize(size(t));
      for (unsigned int m = 0; m < size(t); ++m) {
        std::string type = data.moduleType(moduleLabel(t, m));
        prescalers_[t][m] = (type == "HLTPrescaler");
        moduleTypes_[t][m] = &* moduleTypeSet_.insert(std::move(type)).first;
      }
    }
  }

  std::string const & moduleType(unsigned int trigger, unsigned int module) const {
    return * moduleTypes_.at(trigger).at(module);
  }

  using HLTConfigData::moduleType;

  bool prescaler(unsigned int trigger, unsigned int module) const {
    return prescalers_.at(trigger).at(module);
  }

private:
  std::set<std::string>                         moduleTypeSet_;
  std::vector<std::vector<std::string const*>>  moduleTypes_;
  std::vector<std::vector<bool>>                prescalers_;
};


const char * event_state(bool state) {
  return state ? "accepted" : "rejected";
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
State prescaled_state(int state, int path, int module, HLTConfigDataEx const & config) {
  if (state == Fail and config.prescaler(path, module))
    return Prescaled;
  return (State) state;
}

std::string detailed_path_state(State state, int path, int module, HLTConfigDataEx const & config) {
  auto const & label = config.moduleLabel(path, module);
  auto const & type  = config.moduleType(path, module);

  std::stringstream out;
  out << "'" << path_state(state) << "'";
  if (state == Fail)
    out << " by module " << module << " '" << label << "' [" << type << "]";
  else if (state == Exception)
    out << " at module " << module << " '" << label << "' [" << type << "]";

  return out.str();
}


std::string getProcessNameFromBranch(std::string const & branch) {
  std::vector<boost::iterator_range<std::string::const_iterator>> tokens;
  boost::split(tokens, branch, boost::is_any_of("_."), boost::token_compress_off);
  return boost::copy_range<std::string>(tokens[3]);
}

std::unique_ptr<HLTConfigDataEx> getHLTConfigData(fwlite::EventBase const & event, edm::InputTag inputtag) {
  auto const & history = event.processHistory();
  auto const & branch  = event.getBranchNameFor( edm::Wrapper<edm::TriggerResults>::typeInfo(), inputtag.label().c_str(), inputtag.instance().c_str(), inputtag.process().c_str() );
  auto const & process = getProcessNameFromBranch( branch );
  
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


void compare(std::vector<std::string> const & old_files, edm::InputTag const & old_label, 
             std::vector<std::string> const & new_files, edm::InputTag const & new_label,
             unsigned int max_events, bool ignore_prescales, bool verbose) {

  std::shared_ptr<fwlite::ChainEvent> old_events = std::make_shared<fwlite::ChainEvent>(old_files);
  std::shared_ptr<fwlite::ChainEvent> new_events;

  if (new_files.size() == 1 and new_files[0] == "-")
    new_events = old_events;
  else
    new_events = std::make_shared<fwlite::ChainEvent>(new_files);

  std::unique_ptr<HLTConfigDataEx> old_config;
  std::unique_ptr<HLTConfigDataEx> new_config;

  unsigned int counter = 0;
  bool new_run = true;
  std::vector<TriggerDiff> differences;

  // loop over the reference events
  for (old_events->toBegin(); not old_events->atEnd(); ++(*old_events)) {
    edm::EventID const& id = old_events->id();
    if (new_events != old_events and not new_events->to(id)) {
      std::cerr << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event() << ": not found in the 'new' files, skipping." << std::endl;
      continue;
    }

    fwlite::Handle<edm::TriggerResults> old_handle;
    old_handle.getByLabel<fwlite::Event>(* old_events->event(), old_label.label().c_str(), old_label.instance().c_str(), old_label.process().c_str());
    auto const & old_results = * old_handle;

    fwlite::Handle<edm::TriggerResults> new_handle;
    new_handle.getByLabel<fwlite::Event>(* new_events->event(), new_label.label().c_str(), new_label.instance().c_str(), new_label.process().c_str());
    auto const & new_results = * new_handle;

    if (new_run) {
      new_run = false;
      old_events->fillParameterSetRegistry();
      new_events->fillParameterSetRegistry();

      old_config = getHLTConfigData(* old_events->event(), old_label);
      new_config = getHLTConfigData(* new_events->event(), new_label);
      if (new_config->triggerNames() != old_config->triggerNames()) {
        std::cerr << "Error: inconsistent HLT menus" << std::endl;
        exit(1);
      }

      differences.clear();
      differences.resize(old_config->size());
    }

    bool needs_header = true;
    for (unsigned int p = 0; p < old_config->size(); ++p) {
      State old_state = prescaled_state(old_results.state(p), p, old_results.index(p), * old_config);
      State new_state = prescaled_state(new_results.state(p), p, new_results.index(p), * new_config);

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
        } else if (old_results.index(p) != new_results.index(p)) {
          ++differences[p].internal;
          flag = true;
        }
      }

      if (verbose and flag) {
        if (needs_header) {
          needs_header = false;
          std::cout << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event() << ": "
                    << "old result is '" << event_state(old_results.accept()) << "', "
                    << "new result is '" << event_state(new_results.accept()) << "'"
                    << std::endl;
        }
        std::cout << "    Path " << old_config->triggerName(p) << ":\n"
                  << "        old state is " << detailed_path_state(old_state, p, old_results.index(p), * old_config) << ",\n"
                  << "        new state is " << detailed_path_state(new_state, p, new_results.index(p), * new_config)
                  << std::endl;
      }
    }
    if (verbose and not needs_header)
      std::cout << std::endl;

    ++counter;
    if (max_events and counter >= max_events)
      break;
  }

  std::cout << std::setw(12) << "Events" << std::setw(12) << "Accepted" << std::setw(12) << "Gained" << std::setw(12) << "Lost" << std::setw(12) << "Other" << "  " << "Trigger" << std::endl;
  for (unsigned int p = 0; p < old_config->size(); ++p)
    std::cout << std::setw(12) << counter << differences[p] << "  " << old_config->triggerName(p) << std::endl;
}


int main(int argc, char ** argv) {
  // options
  const char optstring[] = "o:O:n:N:m:pvh";
  const option longopts[] = {
    option{ "old-files",    required_argument,  nullptr, 'o' },
    option{ "old-label",    required_argument,  nullptr, 'O' },
    option{ "new-files",    required_argument,  nullptr, 'n' },
    option{ "new-label",    required_argument,  nullptr, 'N' },
    option{ "max-events",   required_argument,  nullptr, 'm' },
    option{ "prescales",    no_argument,        nullptr, 'p' },
    option{ "verbose",      no_argument,        nullptr, 'v' },
    option{ "help",         no_argument,        nullptr, 'h' },
  };

  // default values
  std::vector<std::string>  old_files;
  edm::InputTag             old_label("TriggerResults");
  std::vector<std::string>  new_files;
  edm::InputTag             new_label("TriggerResults");
  unsigned int              max_events = 0;
  bool                      ignore_prescales = true;
  bool                      verbose = false;

  // parse the command line options
  int c = -1;
  while ((c = getopt_long(argc, argv, optstring, longopts, nullptr)) != -1) {
    switch (c) {
      case 'o':
        old_files.push_back(std::string(optarg));
        while (optind < argc) {
          if (argv[optind][0] == '-')
            break;
          old_files.push_back(std::string(argv[optind]));
          ++optind;
        }
        break;

      case 'O':
        old_label = edm::InputTag(optarg);
        break;

      case 'n':
        new_files.push_back(std::string(optarg));
        while (optind < argc) {
          if (argv[optind][0] == '-')
            break;
          new_files.push_back(std::string(argv[optind]));
          ++optind;
        }
        break;

      case 'N':
        new_label = edm::InputTag(optarg);
        break;

      case 'm':
        max_events = atoi(optarg);
        break;

      case 'p':
        ignore_prescales = false;
        break;

      case 'v':
        verbose = true;
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

  compare(old_files, old_label, new_files, new_label, max_events, ignore_prescales, verbose);

  return 0;
}
