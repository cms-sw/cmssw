// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     edmPluginHelp
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones, W. David Dagenhart
//         Created:  Thu Aug  2 13:33:53 EDT 2007
//

#include "FWCore/PluginManager/interface/PluginInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/validateTopLevelParameterSets.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <boost/program_options.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sys/ioctl.h>
#include <unistd.h>
#include <map>
#include <exception>
#include <utility>

static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kPluginOpt = "plugin";
static char const* const kPluginCommandOpt = "plugin,p";
static char const* const kLibraryOpt = "library";
static char const* const kLibraryCommandOpt = "library,l";
static char const* const kAllLibrariesOpt = "allLibraries";
static char const* const kAllLibrariesCommandOpt = "allLibraries,a";
static char const* const kModuleLabelOpt = "moduleLabel";
static char const* const kModuleLabelCommandOpt = "moduleLabel,m";
static char const* const kBriefOpt = "brief";
static char const* const kBriefCommandOpt = "brief,b";
static char const* const kPrintOnlyLabelsOpt = "printOnlyLabels";
static char const* const kPrintOnlyLabelsCommandOpt = "printOnlyLabels,o";
static char const* const kPrintOnlyPluginsOpt = "printOnlyPlugins";
static char const* const kPrintOnlyPluginsCommandOpt = "printOnlyPlugins,q";
static char const* const kLineWidthOpt = "lineWidth";
static char const* const kLineWidthCommandOpt = "lineWidth,w";
static char const* const kTopLevelOpt = "topLevel";
static char const* const kTopLevelCommandOpt = "topLevel,t";

namespace {

  void getMatchingPlugins(edmplugin::PluginInfo const& pluginInfo,
                          std::vector<edmplugin::PluginInfo>& matchingInfos,
                          std::string& previousName,
                          std::string const& library,
                          std::string const& plugin) {
    // We do not want duplicate plugin names and
    // can safely assume the calls to this function
    // occur with the pluginInfo argument sorted by
    // pluginName
    if (pluginInfo.name_ == previousName)
      return;
    previousName = pluginInfo.name_;

    if (!library.empty() && pluginInfo.loadable_.filename() != library) {
      return;
    }

    if (!plugin.empty() && pluginInfo.name_ != plugin) {
      return;
    }

    matchingInfos.push_back(pluginInfo);
  }

  // -------------------------------------------------------------------

  void writeDocForPlugin(edmplugin::PluginInfo const& pluginInfo,
                         edm::ParameterSetDescriptionFillerPluginFactory* factory,
                         std::string const& moduleLabel,
                         bool brief,
                         bool printOnlyLabels,
                         bool printOnlyPlugins,
                         unsigned lineWidth,
                         int& iPlugin) {
    // Define the output stream for all output
    std::ostream& os = std::cout;
    std::ios::fmtflags oldFlags = os.flags();

    ++iPlugin;

    std::unique_ptr<edm::ParameterSetDescriptionFillerBase> filler;

    try {
      filler = std::unique_ptr<edm::ParameterSetDescriptionFillerBase>{factory->create(pluginInfo.name_)};
    } catch (cms::Exception& e) {
      os << "\nSTART ERROR FROM edmPluginHelp\n";
      os << "The executable \"edmPluginHelp\" encountered a problem while creating a\n"
            "ParameterSetDescriptionFiller, probably related to loading a plugin.\n"
            "This plugin is being skipped.  Here is the info from the exception:\n"
         << e.what() << std::endl;
      os << "END ERROR FROM edmPluginHelp\n\n";
      return;
    }

    std::string const& baseType =
        (filler->extendedBaseType().empty() ? filler->baseType() : filler->extendedBaseType());

    if (printOnlyPlugins) {
      os << std::setfill(' ');
      os << std::left;
      if (iPlugin == 1) {
        os << std::setw(6) << ""
           << " ";
        os << std::setw(50) << "Plugin Name";
        os << std::setw(24) << "Plugin Type";
        os << "Library Name"
           << "\n";
        os << std::setw(6) << ""
           << " ";
        os << std::setw(50) << "-----------";
        os << std::setw(24) << "-----------";
        os << "------------"
           << "\n";
      }
      os << std::setw(6) << iPlugin << " ";
      os << std::setw(50) << pluginInfo.name_;
      os << std::setw(24) << baseType;
      os << pluginInfo.loadable_.filename() << "\n";
      os.flags(oldFlags);
      return;
    }

    os << std::left << iPlugin << "  " << pluginInfo.name_ << "  (" << baseType << ")  "
       << pluginInfo.loadable_.filename() << "\n";
    os.flags(oldFlags);

    edm::ConfigurationDescriptions descriptions(filler->baseType(), pluginInfo.name_);

    try {
      filler->fill(descriptions);
    } catch (cms::Exception& e) {
      os << "\nSTART ERROR FROM edmPluginHelp\n";
      os << "The executable \"edmPluginHelp\" encountered a problem while filling a\n"
            "ParameterSetDescription.  We give up for this plugin and skip printing out\n"
            "this description and any following descriptions for this plugin.  Here\n"
            "is the info from the exception:\n"
         << e.what() << std::endl;
      os << "END ERROR FROM edmPluginHelp\n\n";
      return;
    }

    try {
      int indentation = 0;
      descriptions.print(os, moduleLabel, brief, printOnlyLabels, lineWidth, indentation, iPlugin);
    } catch (cms::Exception& e) {
      os << "\nSTART ERROR FROM edmPluginHelp\n";
      os << "The executable \"edmPluginHelp\" encountered a problem while printing out a\n"
            "ParameterSetDescription.  We give up for this plugin and skip printing out\n"
            "this description and any following descriptions for this plugin.  Here\n"
            "is the info from the exception:\n"
         << e.what() << std::endl;
      os << "END ERROR FROM edmPluginHelp\n\n";
      return;
    }
  }

  void printTopLevelParameterSets(bool brief, size_t lineWidth, std::string const& psetName) {
    std::ostream& os = std::cout;

    edm::ParameterSetDescription description;

    if (psetName == "options") {
      os << "\nDescription of \'options\' top level ParameterSet\n\n";
      edm::fillOptionsDescription(description);

    } else if (psetName == "maxEvents") {
      os << "\nDescription of \'maxEvents\' top level ParameterSet\n\n";
      edm::fillMaxEventsDescription(description);

    } else if (psetName == "maxLuminosityBlocks") {
      os << "\nDescription of \'maxLuminosityBlocks\' top level ParameterSet\n\n";
      edm::fillMaxLuminosityBlocksDescription(description);

    } else if (psetName == "maxSecondsUntilRampdown") {
      os << "\nDescription of \'maxSecondsUntilRampdown\' top level ParameterSet\n\n";
      edm::fillMaxSecondsUntilRampdownDescription(description);
    } else {
      throw cms::Exception("CommandLineArgument")
          << "Unrecognized name for top level parameter set. "
          << "Allowed values are 'options', 'maxEvents', 'maxLuminosityBlocks', and 'maxSecondsUntilRampdown'";
    }

    edm::DocFormatHelper dfh;
    dfh.setBrief(brief);
    dfh.setLineWidth(lineWidth);
    dfh.setSection("1");

    description.print(os, dfh);
    os << "\n";
  }
}  // namespace
// ---------------------------------------------------------------------------------

int main(int argc, char** argv) try {
  // Process the command line arguments
  std::string descString(argv[0]);
  descString += " option [options]\n\n";
  descString += "Prints descriptions of the allowed/required parameters used to\n";
  descString += "configure plugins. Output is ordered by plugin name. Within a\n";
  descString += "plugin, the labels and parameters are ordered based on the order\n";
  descString += "declared by the plugin. Formatted as follows:\n\n";
  descString += "PluginName (PluginType) Library\n";
  descString += "  ModuleLabel\n";
  descString += "    Details of parameters corresponding to this module label\n\n";
  descString += "For more information about the output format see:\n";
  descString += "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideConfigurationValidationAndHelp\n\n";

  descString += "At least one of the following options must be used: -p, -l, -a, -q, or -t\n\n";
  descString += "Allowed options:";
  boost::program_options::options_description desc(descString);

  // clang-format off
  desc.add_options()(kHelpCommandOpt, "produce help message")(
      kPluginCommandOpt, boost::program_options::value<std::string>(), "only print descriptions for this plugin")(
      kLibraryCommandOpt,
      boost::program_options::value<std::string>(),
      "only print descriptions for plugins in this library")(
      kAllLibrariesCommandOpt,
      "allows the program to run without selecting a plugin or library. "
      "This will take a significant amount of time.")(kModuleLabelCommandOpt,
                                                      boost::program_options::value<std::string>(),
                                                      "only print descriptions for this module label")(
      kBriefCommandOpt,
      "do not print comments, more compact format, suppress text"
      " added to help the user understand what the output means")(
      kPrintOnlyLabelsCommandOpt,
      "do not print parameter descriptions, just list module labels matching selection criteria")(
      kPrintOnlyPluginsCommandOpt,
      "do not print parameter descriptions or module labels, just list plugins matching selection criteria")(
      kLineWidthCommandOpt,
      boost::program_options::value<unsigned>(),
      "try to limit lines to this length by inserting newlines between words in comments. Long words or names can "
      "cause the line length to exceed this limit. Defaults to terminal screen width or 80")(
      kTopLevelCommandOpt,
      boost::program_options::value<std::string>(),
      "print only the description for the top level parameter set with this name. Allowed names are 'options', "
      "'maxEvents', 'maxLuminosityBlocks', and 'maxSecondsUntilRampdown'.");
  // clang-format on

  boost::program_options::variables_map vm;
  try {
    store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    notify(vm);
  } catch (boost::program_options::error const& iException) {
    std::cerr << "Exception from command line processing: " << iException.what() << "\n";
    std::cerr << desc << std::endl;
    return 1;
  }

  if (vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  std::string plugin;
  std::string library;
  std::string moduleLabel;
  bool brief = false;
  bool printOnlyLabels = false;
  bool printOnlyPlugins = false;
  std::string printOnlyTopLevel;

  if (vm.count(kPluginOpt)) {
    plugin = vm[kPluginOpt].as<std::string>();
  }
  if (vm.count(kLibraryOpt)) {
    library = vm[kLibraryOpt].as<std::string>();
  }
  if (!vm.count(kAllLibrariesOpt)) {
    if (!vm.count(kPluginOpt) && !vm.count(kLibraryOpt) && !vm.count(kPrintOnlyPluginsOpt) && !vm.count(kTopLevelOpt)) {
      std::cerr << "\nERROR: At least one of the following options must be used: -p, -l, -a, -q, or -t\n\n";
      std::cerr << desc << std::endl;
      return 1;
    }
  }
  if (vm.count(kModuleLabelOpt)) {
    moduleLabel = vm[kModuleLabelOpt].as<std::string>();
  }
  if (vm.count(kBriefOpt)) {
    brief = true;
  }
  if (vm.count(kPrintOnlyLabelsOpt)) {
    printOnlyLabels = true;
  }
  if (vm.count(kPrintOnlyPluginsOpt)) {
    printOnlyPlugins = true;
  }

  unsigned lineWidth = 80U;

  // This next little bit of code was sent to me.  It gets the number
  // of characters per line that show up on the display terminal in
  // use.  It is not standard C++ code and I do not understand sys/ioctl.h
  // very well.  From what I got via google, it should work on any UNIX platform,
  // which as far as I know is all we need to support.  If this code causes
  // problems, then deleting it and just having the default be 80 should
  // work fine.  Or someone could add the appropriate #ifdef for the
  // OS/platform/compiler where this fails.

  if (isatty(2)) {
#ifdef TIOCGWINSZ
    {
      struct winsize w;
      if (ioctl(2, TIOCGWINSZ, &w) == 0) {
        if (w.ws_col > 0)
          lineWidth = w.ws_col;
      }
    }
#else
#ifdef WIOCGETD
    {
      struct uwdata w;
      if (ioctl(2, WIOCGETD, &w) == 0) {
        if (w.uw_width > 0)
          lineWidth = w.uw_width / w.uw_hs;
      }
    }
#endif
#endif
  }

  if (vm.count(kLineWidthOpt)) {
    lineWidth = vm[kLineWidthOpt].as<unsigned>();
  }

  if (vm.count(kTopLevelOpt)) {
    printOnlyTopLevel = vm[kTopLevelOpt].as<std::string>();
    printTopLevelParameterSets(brief, lineWidth, printOnlyTopLevel);
    return 0;
  }

  // Get the list of plugins from the plugin cache

  edm::ParameterSetDescriptionFillerPluginFactory* factory;
  std::vector<edmplugin::PluginInfo> infos;

  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
    typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;
    CatToInfos const& catToInfos = edmplugin::PluginManager::get()->categoryToInfos();
    factory = edm::ParameterSetDescriptionFillerPluginFactory::get();

    CatToInfos::const_iterator itPlugins = catToInfos.find(factory->category());
    if (itPlugins == catToInfos.end()) {
      return 0;
    }
    infos = itPlugins->second;

  } catch (cms::Exception& e) {
    std::cerr << "The executable \"edmPluginHelp\" failed while retrieving the list of parameter description plugins "
                 "from the cache.\n"
              << "The following problem occurred:\n"
              << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "The executable \"edmPluginHelp\" failed while retrieving the list of parameter description plugins "
                 "from the cache.\n"
              << "The following problem occurred:\n"
              << e.what() << std::endl;
    return 1;
  }

  // Select the plugins that match the library and plugin names if
  // any are specified in the command line arguments

  std::vector<edmplugin::PluginInfo> matchingInfos;

  try {
    std::string previousName;

    edm::for_all(infos,
                 std::bind(&getMatchingPlugins,
                           std::placeholders::_1,
                           std::ref(matchingInfos),
                           std::ref(previousName),
                           std::cref(library),
                           std::cref(plugin)));
  } catch (cms::Exception& e) {
    std::cerr << "The executable \"edmPluginHelp\" failed while selecting plugins.\n"
              << "The following problem occurred:\n"
              << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "The executable \"edmPluginHelp\" failed while selecting plugins.\n"
              << "The following problem occurred:\n"
              << e.what() << std::endl;
    return 1;
  }

  // For each selected plugin, fill the ParameterSetDescription for all defined
  // module labels and then print out the details of the description.
  try {
    int iPlugin = 0;

    edm::for_all(matchingInfos,
                 std::bind(&writeDocForPlugin,
                           std::placeholders::_1,
                           factory,
                           std::cref(moduleLabel),
                           brief,
                           printOnlyLabels,
                           printOnlyPlugins,
                           lineWidth,
                           std::ref(iPlugin)));
  } catch (cms::Exception& e) {
    std::cerr << "\nThe executable \"edmPluginHelp\" failed. The following problem occurred:\n"
              << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "\nThe executable \"edmPluginHelp\" failed. The following problem occurred:\n"
              << e.what() << std::endl;
    return 1;
  }

  return 0;
} catch (cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return 1;
} catch (std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
