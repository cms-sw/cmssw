#include <iostream>
#include <string>
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"

int main(int, char **argv)
{
  DDAlgorithm * algo;
  DDCompactView cpv;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::string name("test:DDTestAlgorithm");
  algo = DDAlgorithmFactory::get()->create(name);
  if (algo) {
    algo->execute( cpv );
    std::cout << "OK\n";
  }
  else {
    std::cout << "SEVERE ERROR: algorithm not found in registered plugins!" << std::endl;
    std::cout << "              name=" << name << std::endl;
  }
  return 0;
}

