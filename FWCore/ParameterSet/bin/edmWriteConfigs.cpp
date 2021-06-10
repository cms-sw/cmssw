// -*- C++ -*-
//
// Package:     FWCore/ParameterSet
// Filename:    edmWriteConfigs.cpp
//
// This program takes as input the name of
// a shared object library that is associated
// with a plugin(s).  For each module that is
// defined in that library it determines
// which labels have predefined configurations
// defined in the C++ code of the module.
// For each of these, it fills a ParameterSetDescription
// in memory.  Then it writes a file named
// <module_label>_cfi.py containing the
// default configuration for each of those module
// labels.  It also prints to std::cout
// the name of each of these module labels.
//
// It also does an analogous thing for sources
// and services.  The only difference is that
// each will have at most one description and
// the output filename is <pluginName>_cfi.py.
//
// Original Author:  W. David Dagenhart
//         Created:  10 December 2008

#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/PluginManager/interface/PluginInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"

#include <boost/program_options.hpp>

#include <set>
#include <string>
#include <iostream>
#include <vector>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <utility>
#include "FWCore/Utilities/interface/Signal.h"
#include <sstream>

static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kLibraryOpt = "library";
static char const* const kLibraryCommandOpt = "library,l";
static char const* const kPathOpt = "path";
static char const* const kPathCommandOpt = "path,p";
static char const* const kPluginOpt = "plugin";
static char const* const kPluginCommandOpt = "plugin,x";

namespace {
  void getMatchingPluginNames(edmplugin::PluginInfo const& pluginInfo,
                              std::vector<std::string>& pluginNames,
                              std::string& previousName,
                              std::string const& library) {
    // We do not want duplicate plugin names and
    // can safely assume the calls to this function
    // occur with the pluginInfo argument sorted by
    // pluginName
    if (pluginInfo.name_ == previousName)
      return;
    previousName = pluginInfo.name_;

    // If the library matches save the name
    if (pluginInfo.loadable_.filename() == library) {
      pluginNames.push_back(pluginInfo.name_);
    }
  }

  void writeCfisForPlugin(std::string const& pluginName,
                          edm::ParameterSetDescriptionFillerPluginFactory* factory,
                          std::set<std::string>& usedCfiFileNames) {
    std::unique_ptr<edm::ParameterSetDescriptionFillerBase> filler(factory->create(pluginName));

    std::string baseType = filler->baseType();

    edm::ConfigurationDescriptions descriptions(filler->baseType(), pluginName);

    try {
      edm::convertException::wrap([&]() { filler->fill(descriptions); });
    } catch (cms::Exception& e) {
      std::ostringstream ost;
      ost << "Filling ParameterSetDescriptions for module of base type " << baseType << " with plugin name \'"
          << pluginName << "\'";
      e.addContext(ost.str());
      throw;
    }

    try {
      edm::convertException::wrap([&]() { descriptions.writeCfis(usedCfiFileNames); });
    } catch (cms::Exception& e) {
      std::ostringstream ost;
      ost << "Writing cfi files using ParameterSetDescriptions for module of base type " << baseType
          << " with plugin name \'" << pluginName << "\'";
      e.addContext(ost.str());
      throw;
    }
  }

  struct Listener {
    typedef std::pair<std::string, std::string> NameAndType;
    typedef std::vector<NameAndType> NameAndTypes;

    void newFactory(edmplugin::PluginFactoryBase const* iBase) {
      using std::placeholders::_1;
      using std::placeholders::_2;
      iBase->newPluginAdded_.connect(std::bind(std::mem_fn(&Listener::newPlugin), this, _1, _2));
    }
    void newPlugin(std::string const& iCategory, edmplugin::PluginInfo const& iInfo) {
      nameAndTypes_.push_back(NameAndType(iInfo.name_, iCategory));
    }
    NameAndTypes nameAndTypes_;
  };

  void getPluginsMatchingCategory(Listener::NameAndType const& nameAndType,
                                  std::vector<std::string>& pluginNames,
                                  std::string const& category) {
    if (category == nameAndType.second) {
      pluginNames.push_back(nameAndType.first);
    }
  }
}  // namespace

int main(int argc, char** argv) try {
  edm::FileInPath::disableFileLookup();

  using std::placeholders::_1;
  std::filesystem::path initialWorkingDirectory = std::filesystem::current_path();

  // Process the command line arguments
  std::string descString(argv[0]);
  descString += " [options] [[--";
  descString += kLibraryOpt;
  descString += "] library_filename]\n\n";
  descString += "Generates and writes configuration files that have the suffix _cfi.py.\n";
  descString += "One configuration file is written for each configuration defined with a\n";
  descString += "module label in the fillDescriptions functions of the plugins in the library.\n";
  descString += "Silently does nothing if the library is not in the edmplugincache, does not\n";
  descString += "exist at all, or the plugins in the library have not defined any configurations.\n";
  descString += "Instead of specifying a library, there is also an option to specify a plugin.\n\n";
  descString += "Allowed options";
  boost::program_options::options_description desc(descString);

  // clang-format off
  desc.add_options()(kHelpCommandOpt, "produce help message")(
      kLibraryCommandOpt, boost::program_options::value<std::string>(), "library filename")(
      kPathCommandOpt,
      "When this option is set, the library filename "
      "is interpreted as a relative or absolute path. If there are no directories in "
      "the library filename, then it looks for the library file in the current directory. "
      "Fails with an error message if the path does not lead to a library file that exists "
      "or can be loaded. "
      "If this option is not set, then the library filename should only be "
      "a filename without any directories.  In that case, it is assumed "
      "the build system has already put the library file in the "
      "appropriate place, built the edmplugincache, and the PluginManager "
      "is used to find and load the library.")(
      kPluginCommandOpt,
      boost::program_options::value<std::string>(),
      "plugin name. You must specify either a library or plugin, but not both.");
  // clang-format on

  boost::program_options::positional_options_description p;
  p.add(kLibraryOpt, -1);

  boost::program_options::variables_map vm;
  try {
    store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
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

  std::string library;
  std::string requestedPlugin;

  try {
    edm::convertException::wrap([&]() {
      if (vm.count(kLibraryOpt) && vm.count(kPluginOpt)) {
        throw cms::Exception("Command Line Arguments")
            << "Both library and plugin specified. You must specify one or the other, but not both.";
      }
      if (vm.count(kLibraryOpt)) {
        library = vm[kLibraryOpt].as<std::string>();
      } else if (vm.count(kPluginOpt)) {
        requestedPlugin = vm[kPluginOpt].as<std::string>();
      } else {
        throw cms::Exception("Command Line Arguments")
            << "No library or plugin specified. You must specify one or the other (but not both).";
      }

      edm::ParameterSetDescriptionFillerPluginFactory* factory;
      std::vector<std::string> pluginNames;

      if (vm.count(kPluginOpt)) {
        edmplugin::PluginManager::configure(edmplugin::standard::config());
        factory = edm::ParameterSetDescriptionFillerPluginFactory::get();
        pluginNames.push_back(requestedPlugin);
      }
      // If using the PluginManager to find the library
      else if (!vm.count(kPathOpt)) {
        // From the PluginManager get a reference to a
        // a vector of PlugInInfo's for plugins defining ParameterSetDescriptions.
        // Each PlugInInfo contains the plugin name and library name.
        edmplugin::PluginManager::configure(edmplugin::standard::config());
        typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;
        CatToInfos const& catToInfos = edmplugin::PluginManager::get()->categoryToInfos();
        factory = edm::ParameterSetDescriptionFillerPluginFactory::get();
        CatToInfos::const_iterator itPlugins = catToInfos.find(factory->category());

        // No plugins in this category at all
        if (itPlugins == catToInfos.end())
          return;

        std::vector<edmplugin::PluginInfo> const& infos = itPlugins->second;
        std::string previousName;

        edm::for_all(
            infos,
            std::bind(&getMatchingPluginNames, _1, std::ref(pluginNames), std::ref(previousName), std::cref(library)));

      } else {
        // the library name is part of a path

        Listener listener;
        edmplugin::PluginFactoryManager* pfm = edmplugin::PluginFactoryManager::get();
        pfm->newFactory_.connect(std::bind(std::mem_fn(&Listener::newFactory), &listener, _1));
        edm::for_all(*pfm, std::bind(std::mem_fn(&Listener::newFactory), &listener, _1));

        std::filesystem::path loadableFile(library);

        // If it is just a filename without any directories,
        // then turn it into an absolute path using the current
        // directory when the program starts.  This prevents
        // the function that loads the library from using
        // LD_LIBRARY_PATH or some other location that it searches
        // to find some library that has the same name, but
        // was not the intended target.
        if (loadableFile.filename() == loadableFile.string()) {
          loadableFile = initialWorkingDirectory / loadableFile;
        }

        // This really loads the library into the program
        // The listener records the plugin names and categories as they are loaded
        try {
          edmplugin::SharedLibrary lib(loadableFile);
        } catch (cms::Exception const& iException) {
          if (iException.category() == "PluginLibraryLoadError") {
            std::cerr << "error: edmWriteConfigs caught an exception while loading a plugin library.\n"
                      << "The executable will return success (0) so scram will continue,\n"
                      << "but no cfi files will be written.\n"
                      << iException.what() << std::endl;
            return;
          } else {
            throw;
          }
        }

        // We do not care about PluginCapabilities category so do not bother to try to include them

        factory = edm::ParameterSetDescriptionFillerPluginFactory::get();

        edm::for_all(listener.nameAndTypes_,
                     std::bind(&getPluginsMatchingCategory, _1, std::ref(pluginNames), std::cref(factory->category())));
      }

      std::set<std::string> usedCfiFileNames;
      edm::for_all(pluginNames, std::bind(&writeCfisForPlugin, _1, factory, std::ref(usedCfiFileNames)));
    });
  } catch (cms::Exception& iException) {
    if (!library.empty()) {
      std::ostringstream ost;
      ost << "Processing library " << library;
      iException.addContext(ost.str());
    }
    iException.addContext("Running executable \"edmWriteConfigs\"");
    std::cerr << "----- Begin Fatal Exception " << std::setprecision(0) << edm::TimeOfDay()
              << "-----------------------\n"
              << iException.explainSelf()
              << "----- End Fatal Exception -------------------------------------------------" << std::endl;
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
