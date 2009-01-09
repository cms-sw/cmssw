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
// $Id$

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/PluginManager/interface/PluginInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include "boost/bind.hpp"

#include <string>
#include <iostream>
#include <vector>
#include <exception>
#include <memory>

static char const* const kLibraryOpt = "library";
static char const* const kLibraryCommandOpt = "library,l";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";

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
    filler->fill(descriptions);
    descriptions.writeCfis(baseType, pluginName);
  }
}

int main (int argc, char **argv)
{
  // Process the command line arguments
  std::string descString(argv[0]);
  descString += " [options] [--";
  descString += kLibraryOpt;
  descString += "] library_filename\nAllowed options";
  boost::program_options::options_description desc(descString);   
  desc.add_options()
                  (kHelpCommandOpt, "produce help message")
                  (kLibraryCommandOpt,
                   boost::program_options::value<std::string>(),
                   "library file name");

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

  try {

    std::string library;
    if (vm.count(kLibraryOpt)) {
      library = vm[kLibraryOpt].as<std::string>();
    }
    else {
      throw cms::Exception("Command Line Arguments")
        << "No library specified";
    }

    // From the PluginManager get a reference to a
    // a vector of PlugInInfo's for plugins defining ParameterSetDescriptions.
    // Each PlugInInfo contains the plugin name and library name.
    edmplugin::PluginManager::configure(edmplugin::standard::config());
    typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;
    CatToInfos const& catToInfos 
      = edmplugin::PluginManager::get()->categoryToInfos();
    edm::ParameterSetDescriptionFillerPluginFactory* factory =
      edm::ParameterSetDescriptionFillerPluginFactory::get();
    CatToInfos::const_iterator itPlugins = catToInfos.find(factory->category());

    // No plugins in this category at all
    if(itPlugins == catToInfos.end() ) return 0;

    std::vector<edmplugin::PluginInfo> const& infos = itPlugins->second;
    std::vector<std::string> pluginNames;
    std::string previousName;

    edm::for_all(infos, boost::bind(&getMatchingPluginNames,
                                    _1,
                                    boost::ref(pluginNames),
                                    boost::ref(previousName),
                                    boost::cref(library)));

    edm::for_all(pluginNames, boost::bind(&writeCfisForPlugin,
                                          _1,
                                          factory));
  } catch(cms::Exception& e) {
    std::cerr << "The following problem occurred\n" 
              << e.what() << std::endl;
    return 1;
  } catch(const std::exception& e) {
    std::cerr << "The following problem occurred\n"
              << e.what() << std::endl;
    return 1;
  }
  return 0;
}
