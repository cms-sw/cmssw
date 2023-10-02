#include "FWCore/Framework/interface/CmsRunParser.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <vector>
#include <string>
#include <iostream>

namespace {
  //additional_parser is documented at https://www.boost.org/doc/libs/1_83_0/doc/html/program_options/howto.html#id-1.3.30.6.3
  //extra_style_parser was added in https://www.boost.org/users/history/version_1_33_0.html
  //it allows performing a fully custom processing on whatever arguments have not yet been processed
  //the function is responsible for removing processed arguments from the input vector
  //and assembling the return value, which is a vector of boost's option type
  //some usage examples: https://stackoverflow.com/a/5481228, https://stackoverflow.com/a/37993517
  //internally, when cmdline::run() is called, the function given to extra_style_parser is added to the beginning of list of parsers
  //all parsers are called in order until args is empty (validity of non-empty output is checked after each parser)
  std::vector<boost::program_options::option> final_opts_parser(std::vector<std::string>& args) {
    std::vector<boost::program_options::option> result;
    std::string configName;
    if (!args.empty() and !args[0].empty()) {
      if (args[0][0] != '-') {  // name is first positional arg -> doesn't start with '-'
        configName = args[0];
        args.erase(args.begin());
      } else if (args[0] == "--" and args.size() > 1) {  // name can start with '-' if separator comes first
        configName = args[1];
        args.erase(args.begin(), args.begin() + 2);
      }
    }
    if (!configName.empty()) {
      result.emplace_back(edm::CmsRunParser::kParameterSetOpt, std::vector<std::string>(1, configName));
      result.emplace_back();
      auto& pythonOpts = result.back();
      pythonOpts.string_key = edm::CmsRunParser::kPythonOpt;
      pythonOpts.value.reserve(args.size());
      pythonOpts.original_tokens.reserve(args.size());
      for (const auto& arg : args) {
        pythonOpts.value.push_back(arg);
        pythonOpts.original_tokens.push_back(arg);
      }
      //default value to avoid "is missing" error
      if (pythonOpts.value.empty()) {
        pythonOpts.value.push_back(edm::CmsRunParser::kPythonOptDefault);
        pythonOpts.original_tokens.push_back("");
      }
      args.clear();
    }
    return result;
  }
}  // namespace

namespace edm {
  CmsRunParser::CmsRunParser(const char* name)
      : desc_(std::string(name) + " [options] [--] config_file [python options]\nAllowed options"),
        all_options_("All Options") {
    // clang-format off
      desc_.add_options()(kHelpCommandOpt, "produce help message")(
          kJobreportCommandOpt,
          boost::program_options::value<std::string>(),
          "file name to use for a job report file: default extension is .xml")(
          kEnableJobreportCommandOpt, "enable job report files (if any) specified in configuration file")(
          kJobModeCommandOpt,
          boost::program_options::value<std::string>(),
          "Job Mode for MessageLogger defaults - default mode is grid")(
          kNumberOfThreadsCommandOpt,
          boost::program_options::value<unsigned int>(),
          "Number of threads to use in job (0 is use all CPUs)")(
          kSizeOfStackForThreadCommandOpt,
          boost::program_options::value<unsigned int>(),
          "Size of stack in KB to use for extra threads (0 is use system default size)")(kStrictOpt, "strict parsing")(
          kCmdCommandOpt, boost::program_options::value<std::string>(), "config passed in as string (cannot be used with config_file)");
    // clang-format on

    // anything at the end will be ignored, and sent to python
    pos_options_.add(kParameterSetOpt, 1).add(kPythonOpt, -1);

    // This --fwk option is not used anymore, but I'm leaving it around as
    // it might be useful again in the future for code development
    // purposes.  We originally used it when implementing the boost
    // state machine code.
    boost::program_options::options_description hidden("hidden options");
    hidden.add_options()("fwk", "For use only by Framework Developers")(
        kParameterSetOpt, boost::program_options::value<std::string>(), "configuration file")(
        kPythonOpt,
        boost::program_options::value<std::vector<std::string>>(),
        "options at the end to be passed to python");

    all_options_.add(desc_).add(hidden);
  }
  CmsRunParser::MapOrExit CmsRunParser::parse(int argc, char* argv[]) const {
    boost::program_options::variables_map vm;
    try {
      store(boost::program_options::command_line_parser(argc, argv)
                .extra_style_parser(final_opts_parser)
                .options(all_options_)
                .positional(pos_options_)
                .run(),
            vm);
      notify(vm);
    } catch (boost::program_options::error const& iException) {
      edm::LogAbsolute("CommandLineProcessing")
          << "cmsRun: Error while trying to process command line arguments:\n"
          << iException.what() << "\nFor usage and an options list, please do 'cmsRun --help'.";
      return MapOrExit(edm::errors::CommandLineProcessing);
    }

    if (vm.count(kHelpOpt)) {
      std::cout << desc_ << std::endl;
      edm::HaltMessageLogging();
      return MapOrExit(0);
    }

    return MapOrExit(vm);
  }
}  // namespace edm
