
#include "oneTimeInitialization.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"

namespace {

  bool oneTimeInitializationImpl() {
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    static std::unique_ptr<edm::ThreadsController> tsiPtr = std::make_unique<edm::ThreadsController>(1);

    // register the empty parentage vector , once and for all
    edm::ParentageRegistry::instance()->insertMapped(edm::Parentage());

    // register the empty parameter set, once and for all.
    edm::ParameterSet().registerIt();
    return true;
  }
}  //namespace

namespace edm::testprocessor {
  bool oneTimeInitialization() {
    static const bool s_init{oneTimeInitializationImpl()};
    return s_init;
  }
}  // namespace edm::testprocessor
