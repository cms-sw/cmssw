#include "CondCore/ESSources/interface/ProductResolverFactory.h"
#include "CondCore/ESSources/interface/ProductResolver.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include <iostream>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "requires exactly one argument";
    return 1;
  }

  std::string argument(argv[1]);
  if (argument == "-h" or argument == "--help") {
    std::cout << "condRecordToDataProduct [record]\n\n"
                 "returns the C++ type name of the DataProduct held by Record [record]"
              << std::endl;
    return 0;
  }

  edmplugin::PluginManager::configure(edmplugin::standard::config());

  auto plugin = cond::ProductResolverFactory::get()->tryToCreate(argument + "@NewProxy");
  if (not plugin) {
    std::cerr << "unable to find proxy for record " << argument;
    return 2;
  }

  std::cout << plugin->type().name();
  return 0;
}
