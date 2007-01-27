#include <iostream>
#include <string>
#include "PluginManager/PluginManager.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "SealBase/Signal.h"

int main(int, char **argv)
{
  seal::Signal::handleFatal (argv [0]);
  DDAlgorithm * algo;
  PluginManager::get()->initialise();
  string name("DDTestAlgorithm");
  algo = DDAlgorithmFactory::get()->create(name);
  if (algo) {
    algo->execute();
  }
  else {
    cout << "SEVERE ERROR: algorithm not found in registered plugins!" << endl;
    cout << "              name=" << name << endl;
  }
  return 0;
}
