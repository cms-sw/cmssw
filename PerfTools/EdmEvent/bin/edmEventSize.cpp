/** measure branch or leaves sizes
 *
 *
 */

#include "PerfTools/EdmEvent/interface/EdmEventSize.h"

#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <fstream>

#include <TROOT.h>
#include <TSystem.h>
#include <TError.h>
#include "FWCore/FWLite/interface/FWLiteEnabler.h"

static const char* const kHelpOpt = "help";
static const char* const kHelpCommandOpt = "help,h";
static const char* const kDataFileOpt = "data-file";
static const char* const kDataFileCommandOpt = "data-file,d";
static const char* const kTreeNameOpt = "tree-name";
static const char* const kTreeNameCommandOpt = "tree-name,n";
static const char* const kOutputOpt = "output";
static const char* const kOutputCommandOpt = "output,o";
static const char* const kAutoLoadOpt = "auto-loader";
static const char* const kAutoLoadCommandOpt = "auto-loader,a";
static const char* const kPlotOpt = "plot";
static const char* const kPlotCommandOpt = "plot,p";
static const char* const kSavePlotOpt = "save-plot";
static const char* const kSavePlotCommandOpt = "save-plot,s";
static const char* const kPlotTopOpt = "plot-top";
static const char* const kPlotTopCommandOpt = "plot-top,t";
static const char* const kVerboseOpt = "verbose";
static const char* const kVerboseCommandOpt = "verbose,v";
static const char* const kAlphabeticOrderOpt = "alphabetic-order";
static const char* const kAlphabeticOrderCommandOpt = "alphabetic-order,A";
static const char* const kFormatNamesOpt = "format-names";
static const char* const kFormatNamesCommandOpt = "format-names,F";
static const char* const kOutputFormatOpt = "output-format";
static const char* const kOutputFormatCommandOpt = "output-format,f";
static const char* const kLeavesSizeOpt = "get-leaves-size";
static const char* const kLeavesSizeCommandOpt = "get-leaves-size,L";

template <perftools::EdmEventMode M>
void processRecord(const boost::program_options::variables_map& vm,
                   const std::string& fileName,
                   const std::string& treeName,
                   bool verbose) {
  using EventSize = typename perftools::EdmEventSize<M>;
  using Error = typename EventSize::Error;
  try {
    EventSize me(fileName, treeName);

    if (vm.count(kFormatNamesOpt))
      me.formatNames();

    if (vm.count(kAlphabeticOrderOpt))
      me.sortAlpha();

    if (verbose) {
      std::cout << std::endl;
      if (vm.count(kOutputFormatOpt) && vm[kOutputFormatOpt].as<std::string>() == "json")
        me.dumpJson(std::cout);
      else
        me.dump(std::cout);
      std::cout << std::endl;
    }

    if (vm.count(kOutputOpt)) {
      std::ofstream of(vm[kOutputOpt].as<std::string>().c_str());
      if (vm.count(kOutputFormatOpt) && vm[kOutputFormatOpt].as<std::string>() == "json")
        me.dumpJson(of);
      else
        me.dump(of);
      of << std::endl;
    }

    bool plot = (vm.count(kPlotOpt) > 0);
    bool save = (vm.count(kSavePlotOpt) > 0);
    if (plot || save) {
      std::string plotName;
      std::string histName;
      if (plot)
        plotName = vm[kPlotOpt].as<std::string>();
      if (save)
        histName = vm[kSavePlotOpt].as<std::string>();
      int top = 0;
      if (vm.count(kPlotTopOpt) > 0)
        top = vm[kPlotTopOpt].as<int>();
      me.produceHistos(plotName, histName, top);
    }
  } catch (Error const& error) {
    std::cerr << "Error: " << error.descr << std::endl;
    exit(error.code);
  }
}

int main(int argc, char* argv[]) {
  using namespace boost::program_options;

  std::string programName(argv[0]);
  std::string descString(programName);
  descString += " [options] ";
  descString += "data_file \nAllowed options";
  options_description desc(descString);

  // clang-format off
    desc.add_options()(kHelpCommandOpt, "produce help message")(
            kAutoLoadCommandOpt,"automatic library loading (avoid root warnings)")(
            kDataFileCommandOpt, value<std::string>(), "data file")(
            kTreeNameCommandOpt, value<std::string>(), "tree name (default \"Events\")")(
            kOutputCommandOpt, value<std::string>(), "output file")(
            kAlphabeticOrderCommandOpt, "sort by alphabetic order (default: sort by size)")(
            kFormatNamesCommandOpt, "format product name as \"product:label (type)\" or \"product:label (type) object (object type)\" (default: use full branch name)")(
            kPlotCommandOpt, value<std::string>(), "produce a summary plot")(
            kPlotTopCommandOpt, value<int>(), "plot only the <arg> top size branches")(
            kSavePlotCommandOpt, value<std::string>(), "save plot into root file <arg>")(
            kVerboseCommandOpt, "verbose printout")(
            kOutputFormatCommandOpt, value<std::string>(), "output file format as text or json (default: text)")(
            kLeavesSizeCommandOpt, "get size of every leaf in the tree");
  // clang-format on

  positional_options_description p;

  p.add(kDataFileOpt, -1);

  variables_map vm;
  try {
    store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    notify(vm);
  } catch (const error&) {
    return 7000;
  }

  if (vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count(kDataFileOpt)) {
    std::cerr << programName << ": no data file given" << std::endl;
    return 7001;
  }

  gROOT->SetBatch();

  if (vm.count(kAutoLoadOpt) != 0) {
    gSystem->Load("libFWCoreFWLite");
    FWLiteEnabler::enable();
  } else
    gErrorIgnoreLevel = kError;

  bool verbose = vm.count(kVerboseOpt) > 0;

  std::string fileName = vm[kDataFileOpt].as<std::string>();

  std::string treeName = "Events";
  if (vm.count(kTreeNameOpt))
    treeName = vm[kTreeNameOpt].as<std::string>();

  if (vm.count(kLeavesSizeOpt))
    processRecord<perftools::EdmEventMode::Leaves>(vm, fileName, treeName, verbose);
  else
    processRecord<perftools::EdmEventMode::Branches>(vm, fileName, treeName, verbose);

  return 0;
}
