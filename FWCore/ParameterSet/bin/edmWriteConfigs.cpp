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
// Original Author:  W. David Dagenhart
//         Created:  10 December 2008

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/PluginManager/interface/PluginInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"

#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/bind.hpp>
#include <boost/mem_fn.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <exception>
#include <memory>
#include <utility>
#include "sigc++/signal.h"

static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kLibraryOpt = "library";
static char const* const kLibraryCommandOpt = "library,l";
static char const* const kPathOpt = "path";
static char const* const kPathCommandOpt = "path,p";

namespace {
  void getMatchingPluginNames(edmplugin::PluginInfo const& pluginInfo,
                              std::vector<std::string> & pluginNames,
                              std::string & previousName,
                              std::string const& library) {
    // We do not want duplicate plugin names and
    // can safely assume the calls to this function
    // occur with the pluginInfo argument sorted by
    // pluginName
    if (pluginInfo.name_ == previousName) return;
    previousName = pluginInfo.name_;

    // If the library matches save the name
    if (pluginInfo.loadable_.leaf() == library) {
      pluginNames.push_back(pluginInfo.name_);
    }
  }

  void writeCfisForPlugin(std::string const& pluginName,
                          edm::ParameterSetDescriptionFillerPluginFactory* factory)
  {
    std::auto_ptr<edm::ParameterSetDescriptionFillerBase> filler(factory->create(pluginName));

    std::string baseType = filler->baseType();
 
    edm::ConfigurationDescriptions descriptions;

    try {
      filler->fill(descriptions);
    }
    catch(cms::Exception& e) {
      edm::Exception toThrow(edm::errors::LogicError,
        "Error occurred while creating ParameterSetDescriptions\n");
        toThrow << "for module of base type \'" << baseType << "\' with plugin name \'" << pluginName << "\'\n";
        toThrow.append(e);
        throw toThrow;
    }

    try {
      descriptions.writeCfis(baseType, pluginName);
    }
    catch(cms::Exception& e) {
      edm::Exception toThrow(edm::errors::LogicError,
        "Error occurred while writing cfi files using ParameterSetDescriptions\n");
        toThrow << "for module of base type \'" << baseType << "\' with plugin name \'" << pluginName << "\'\n";
        toThrow.append(e);
        throw toThrow;
    }
  }

  struct Listener {

    typedef std::pair< std::string, std::string> NameAndType;
    typedef std::vector< NameAndType > NameAndTypes;

    void newFactory(const edmplugin::PluginFactoryBase* iBase) {
      iBase->newPluginAdded_.connect(boost::bind(boost::mem_fn(&Listener::newPlugin),this,_1,_2));
    }
    void newPlugin(const std::string& iCategory, const edmplugin::PluginInfo& iInfo) {
      nameAndTypes_.push_back(NameAndType(iInfo.name_,iCategory));
    }    
    NameAndTypes nameAndTypes_;
  };

  void getPluginsMatchingCategory(Listener::NameAndType const& nameAndType,
                                  std::vector<std::string> & pluginNames,
                                  std::string const& category) {
    if (category == nameAndType.second){
      pluginNames.push_back(nameAndType.first);
    }
  }
}

int main (int argc, char **argv)
{
  boost::filesystem::path initialWorkingDirectory =
    boost::filesystem::initial_path<boost::filesystem::path>();

  // Process the command line arguments
  std::string descString(argv[0]);
  descString += " [options] [--";
  descString += kLibraryOpt;
  descString += "] library_filename\n\n";
  descString += "Generates and writes configuration files which have the suffix _cfi.py.\n";
  descString += "One configuration file is written for each configuration defined with a\n";
  descString += "module label in the fillDescriptions functions of the plugins in the library.\n";
  descString += "Silently does nothing if the library is not in the edmplugincache, does not\n";
  descString += "exist at all, or the plugins in the library have not defined any configurations.\n\n";
  descString += "Allowed options";
  boost::program_options::options_description desc(descString);   
  desc.add_options()
                  (kHelpCommandOpt, "produce help message")
                  (kLibraryCommandOpt,
                   boost::program_options::value<std::string>(),
                   "library filename")
                  (kPathCommandOpt, "When this option is set, the library filename "
                   "is interpreted as a relative or absolute path. If there are no directories in "
                   "the library filename, then it looks for the library file in the current directory. "
                   "Fails with an error message if the path does not lead to a library file that exists "
                   "or can be loaded. "
                   "If this option is not set, then the library filename should only be "
                   "a filename without any directories.  In that case, it is assumed "
                   "the build system has already put the library file in the "
                   "appropriate place, built the edmplugincache, and the PluginManager "
                   "is used to find and load the library.");

  boost::program_options::positional_options_description p;
  p.add(kLibraryOpt, -1);

  boost::program_options::variables_map vm;
  try {
    store(boost::program_options::command_line_parser(argc,argv).options(desc).positional(p).run(),vm);
    notify(vm);
  } catch(boost::program_options::error const& iException) {
    std::cerr << "Exception from command line processing: "
              << iException.what() << "\n";
    std::cerr << desc << std::endl;
    return 1;
  }     

  if(vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  std::string library;

  try {

    if (vm.count(kLibraryOpt)) {
      library = vm[kLibraryOpt].as<std::string>();
    }
    else {
      throw cms::Exception("Command Line Arguments")
        << "No library specified";
    }

    edm::ParameterSetDescriptionFillerPluginFactory* factory;
    std::vector<std::string> pluginNames;

    // If using the PluginManager to find the library
    if(!vm.count(kPathOpt)) {

      // From the PluginManager get a reference to a
      // a vector of PlugInInfo's for plugins defining ParameterSetDescriptions.
      // Each PlugInInfo contains the plugin name and library name.
      edmplugin::PluginManager::configure(edmplugin::standard::config());
      typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;
      CatToInfos const& catToInfos 
        = edmplugin::PluginManager::get()->categoryToInfos();
      factory =
        edm::ParameterSetDescriptionFillerPluginFactory::get();
      CatToInfos::const_iterator itPlugins = catToInfos.find(factory->category());

      // No plugins in this category at all
      if(itPlugins == catToInfos.end() ) return 0;

      std::vector<edmplugin::PluginInfo> const& infos = itPlugins->second;
      std::string previousName;

      edm::for_all(infos, boost::bind(&getMatchingPluginNames,
                                      _1,
                                      boost::ref(pluginNames),
                                      boost::ref(previousName),
                                      boost::cref(library)));

    }
    // the library name is part of a path
    else {

      Listener listener;
      edmplugin::PluginFactoryManager* pfm = edmplugin::PluginFactoryManager::get();
      pfm->newFactory_.connect(boost::bind(boost::mem_fn(&Listener::newFactory),&listener,_1));
      edm::for_all(*pfm, boost::bind(boost::mem_fn(&Listener::newFactory),&listener,_1));

      boost::filesystem::path loadableFile(library);

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
      edmplugin::SharedLibrary lib(loadableFile);

      // We do not care about PluginCapabilities category so do not bother to try to include them

      factory =
        edm::ParameterSetDescriptionFillerPluginFactory::get();

      edm::for_all(listener.nameAndTypes_, boost::bind(&getPluginsMatchingCategory,
                                                       _1,
                                                       boost::ref(pluginNames),
                                                       boost::cref(factory->category())));
    }

    edm::for_all(pluginNames, boost::bind(&writeCfisForPlugin,
                                          _1,
                                          factory));

  } catch(cms::Exception& e) {
    std::cerr << "The executable \"edmWriteConfigs\" failed while processing library " << library << ".\n"
              << "The following problem occurred:\n" 
              << e.what() << std::endl;
    return 1;
  } catch(const std::exception& e) {
    std::cerr << "The executable \"edmWriteConfigs\" failed while processing library " << library << ".\n"
              << "The following problem occurred:\n" 
              << e.what() << std::endl;
    return 1;
  }
  return 0;
}
