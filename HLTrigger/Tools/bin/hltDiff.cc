/*      hltDiff: compare TriggerResults event by event
 *
 *      Compare two TriggerResults collections event by event.
 *      These can come from two collections in the same file(s), or from two different sets of files.
 */

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <memory>

#include <cstring>
#include <unistd.h>
#include <getopt.h>

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"

void usage(std::ostream & out) {
  out << "\
usage: hltDiff -o|--old-files FILE1.ROOT [FILE2.ROOT ...] [-O|--old-label LABEL[:INSTANCE[:PROCESS]]]\n\
               -n|--new-files FILE1.ROOT [FILE2.ROOT ...] [-N|--new-label LABEL[:INSTANCE[:PROCESS]]]\n\
               [-m|--max-events MAXEVENTS] [-v|--verbose] [-h|--help]\n\
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

const char * decode_path_status(int status) {
  static const char * message[] = { "not run", "accepted", "rejected", "error" };

  if (status > 0 and status < 4)
    return message[status];
  else
    return "invalid";
}
  
/*
def build_menu(event, results):
    tn = event.triggerNames(results)
    names   = [ tn.triggerName(i) for i in range(results.size()) ]
*/

struct TriggerDiff {
  TriggerDiff() : count(0), gained(0), lost(0) { }
    
  unsigned int count;
  unsigned int gained;
  unsigned int lost;

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
      << std::setw(12) << TriggerDiff::format(diff.lost, '-');
  return out;
}



void compare(std::vector<std::string> const & old_files, edm::InputTag const & old_label, 
             std::vector<std::string> const & new_files, edm::InputTag const & new_label,
             unsigned int max_events, bool verbose) {

  std::shared_ptr<fwlite::ChainEvent> old_events_p = std::make_shared<fwlite::ChainEvent>(old_files);
  std::shared_ptr<fwlite::ChainEvent> new_events_p;
  bool same_files;

  if (new_files.size() == 1 and new_files[0] == "-") {
    new_events_p = old_events_p;
    same_files = true;
  } else {
    new_events_p = std::make_shared<fwlite::ChainEvent>(new_files);
    same_files = false;
  }

  fwlite::ChainEvent & old_events = * old_events_p;
  fwlite::ChainEvent & new_events = * new_events_p;

  unsigned int counter = 0;
  bool new_run = true;
  std::vector<std::string> const * trigger_names = nullptr;
  std::vector<TriggerDiff> differences;

  // loop over the reference events
  for (old_events.toBegin(); not old_events.atEnd(); ++old_events) {
    edm::EventID const& id = old_events.id();
    if (not same_files and not new_events.to(id)) {
      std::cerr << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event() << ": not found in the 'new' files, skipping." << std::endl;
      continue;
    }

    fwlite::Handle<edm::TriggerResults> old_handle;
    old_handle.getByLabel<fwlite::Event>(* old_events.event(), old_label.label().c_str(), old_label.instance().c_str(), old_label.process().c_str());
    auto const & old_results = * old_handle;

    fwlite::Handle<edm::TriggerResults> new_handle;
    new_handle.getByLabel<fwlite::Event>(* new_events.event(), new_label.label().c_str(), new_label.instance().c_str(), new_label.process().c_str());
    auto const & new_results = * new_handle;

    if (new_run) {
      new_run = false;
      trigger_names = & old_events.triggerNames(old_results).triggerNames();
      if (new_events.triggerNames(new_results).triggerNames() != * trigger_names) {
        std::cerr << "Error: inconsistent HLT menus" << std::endl;
        return;
      }
      differences.clear();
      differences.resize(trigger_names->size());
    }

    bool needs_header = true;
    for (unsigned int p = 0; p < trigger_names->size(); ++p) {
      if (old_results.state(p) == edm::hlt::Pass)
        ++differences[p].count;
      if (old_results.state(p) == edm::hlt::Pass and new_results.state(p) != edm::hlt::Pass)
        ++differences[p].lost;
      else if (old_results.state(p) != edm::hlt::Pass and new_results.state(p) == edm::hlt::Pass)
        ++differences[p].gained;

      if (verbose) {
        if (old_results.state(p) != new_results.state(p) or old_results.index(p) != new_results.index(p)) {
          if (needs_header) {
            needs_header = false;
            std::cout << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event() << ": old result " << old_results.accept() << " , new result " << new_results.accept() << std::endl;
          }
          std::cout << "  Path " << (*trigger_names)[p] << ": "
                    << "  old state is '" << decode_path_status(old_results.state(p)) << "' due to module " << old_results.index(p)
                    << "  new state is '" << decode_path_status(new_results.state(p)) << "' due to module " << new_results.index(p)
                    << std::endl;
        }
      }
    }
    if (not needs_header)
      std::cout << std::endl;

    ++counter;
    if (max_events and counter >= max_events)
      break;
  }

  std::cout << std::setw(12) << "Events" << std::setw(12) << "Accepted" << std::setw(12) << "Gained" << std::setw(12) << "Lost" << "  " << "Trigger" << std::endl;
  for (unsigned int p = 0; p < trigger_names->size(); ++p)
    std::cout << std::setw(12) << counter << differences[p] << "  " << (*trigger_names)[p] << std::endl;
}


int main(int argc, char ** argv) {
  // options
  const char optstring[] = "o:O:n:N:m:vh";
  const option longopts[] = {
    option{ "old-files",    required_argument,  nullptr, 'o' },
    option{ "old-label",    required_argument,  nullptr, 'O' },
    option{ "new-files",    required_argument,  nullptr, 'n' },
    option{ "new-label",    required_argument,  nullptr, 'N' },
    option{ "max-events",   required_argument,  nullptr, 'm' },
    option{ "verbose",      no_argument,        nullptr, 'v' },
    option{ "help",         no_argument,        nullptr, 'h' },
  };

  // default values
  std::vector<std::string>  old_files;
  edm::InputTag             old_label("TriggerResults");
  std::vector<std::string>  new_files;
  edm::InputTag             new_label("TriggerResults");
  unsigned int              max_events = 0;
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

  if (old_files.empty() or new_files.empty()) {
    error(std::cerr, "hltDiff: missing file operand");
    exit(1);
  }

  compare(old_files, old_label, new_files, new_label, max_events, verbose);

  return 0;
}
