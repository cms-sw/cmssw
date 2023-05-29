#include <cstdlib>

#include <iostream>
#include <iomanip>
#include "boost/filesystem.hpp"
#include "boost/version.hpp"
#include "boost/program_options.hpp"

#include "Options.h"

using namespace std;
namespace fs = boost::filesystem;
namespace po = boost::program_options;  // used to read the options from command line

namespace AllInOneConfig {

  void check_file(string file) {
    // check that the config file exists
    if (!fs::exists(file)) {
      cerr << "Didn't find configuration file " << file << '\n';
      exit(EXIT_FAILURE);
    }
  }

  void set_verbose(bool flag) {
    if (!flag)
      cout.setstate(ios_base::failbit);
  }

  void set_silent(bool flag) {
    if (flag)
      cerr.setstate(ios_base::failbit);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Constructor for PO:
  /// - contains a parser for the help itself
  /// - contains a parser for the options (unless help was displayed)
  /// - and contains a hidden/position option to get the JSON config file
  Options::Options(bool getter) : help{"Helper"}, desc{"Options"}, hide{"Hidden"} {
    // define all options
    help.add_options()("help,h", "Help screen")("example,e", "Print example of config in JSON format")(
        "tutorial,t", "Explain how to use the command");

    // then (except for getJSON) the running options
    if (!getter)
      desc.add_options()(
          "dry,d", po::bool_switch(&dry)->default_value(false), "Set up everything, but don't run anything")(
          "verbose,v", po::bool_switch()->default_value(false)->notifier(set_verbose), "Enable standard output stream")(
          "silent,s", po::bool_switch()->default_value(false)->notifier(set_silent), "Disable standard error stream");

    // and finally / most importantly, the config file as apositional argument
    hide.add_options()("config,c", po::value<string>(&config)->required()->notifier(check_file), "JSON config file");
    pos_hide.add("config", 1);  // 1 means one (positional) argument
    if (getter) {
      // in case of getIMFO, adding a second positional argument for the key
      hide.add_options()("key,k", po::value<string>(&key)->required(), "key");
      pos_hide.add("key", 1);
    }
  }

  void Options::helper(int argc, char* argv[]) {
    po::options_description options;
    options.add(help).add(desc);

    po::command_line_parser parser_helper{argc, argv};
    parser_helper
        .options(help)          // don't parse required options before parsing help
        .allow_unregistered();  // ignore unregistered options,

    // defines actions
    po::variables_map vm;
    po::store(parser_helper.run(), vm);
    po::notify(vm);  // necessary for config to be given the value from the cmd line

    if (vm.count("help")) {
      fs::path executable = argv[0];
      cout << "Basic syntax:\n  " << executable.filename().string() << " config.JSON\n"
           << options << '\n'
           << "Boost " << BOOST_LIB_VERSION << endl;
      exit(EXIT_SUCCESS);
    }

    if (vm.count("example")) {
      static const char *bold = "\e[1m", *normal = "\e[0m";
      cout << bold << "Example of HTC condor config:" << normal << '\n' << endl;
      system("cat $CMSSW_BASE/src/Alignment/OfflineValidation/bin/.default.sub");
      cout << '\n' << bold << "Example of JSON config (for `validateAlignment` only):" << normal << '\n' << endl;
      system("cat $CMSSW_BASE/src/Alignment/OfflineValidation/bin/.example.JSON");
      cout << '\n'
           << bold << "NB: " << normal
           << " for examples of inputs to GCPs, DMRs, etc., just make a dry run of the example" << endl;
      exit(EXIT_SUCCESS);
    }
    // TODO: make examples for each executables? (could be examples used in release validation...)

    if (vm.count("tutorial")) {
      cout << "   ______________________\n"
           << "  < Oops! not ready yet! >\n"
           << "   ----------------------\n"
           << "          \\   ^__^\n"
           << "           \\  (oo)\\_______\n"
           << "              (__)\\       )\\/\\\n"
           << "                  ||----w |\n"
           << "                  ||     ||\n"
           << flush;
      exit(EXIT_SUCCESS);
    }
  }

  void Options::parser(int argc, char* argv[]) {
    po::options_description cmd_line;
    cmd_line.add(desc).add(hide);

    po::command_line_parser parser{argc, argv};
    parser.options(cmd_line).positional(pos_hide);

    po::variables_map vm;
    po::store(parser.run(), vm);
    po::notify(vm);  // necessary for config to be given the value from the cmd line
  }

}  // end of namespace AllInOneConfig
