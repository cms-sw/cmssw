#include <iostream>
#include <string>

#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

int main(int, char **argv)
{
  DDAlgorithm * algo;
  DDCompactView cpv;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::string name("test:DDTestAlgorithm");
  algo = DDAlgorithmFactory::get()->create(name);
  if (algo) {
    const DDNumericArguments nArgs;
    const DDVectorArguments vArgs;
    const DDMapArguments mArgs;
    const DDStringArguments sArgs;
    const DDStringVectorArguments vsArgs;
    algo->initialize( nArgs, vArgs, mArgs, sArgs, vsArgs );
    algo->execute( cpv );
    std::cout << "OK\n";
  }
  else {
    std::cout << "SEVERE ERROR: algorithm not found in registered plugins!" << std::endl;
    std::cout << "              name=" << name << std::endl;
  }
  return 0;
}

