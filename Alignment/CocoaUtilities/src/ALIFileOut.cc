//   COCOA class implementation file
//Id:  ALIFileOut.C
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"

#include <cstdlib>
#include <strstream>

std::vector<ALIFileOut*> ALIFileOut::theInstances;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ get the instance of file with name filename
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIFileOut& ALIFileOut::getInstance(const ALIstring& filename) {
  std::vector<ALIFileOut*>::const_iterator vfcite;
  for (vfcite = theInstances.begin(); vfcite != theInstances.end(); ++vfcite) {
    if ((*vfcite)->name() == filename) {
      return *(*vfcite);
      break;
    }
  }

  if (vfcite == theInstances.end()) {
    ALIFileOut* instance = new ALIFileOut(filename);
    instance->open(filename.c_str());
    if (!instance) {
      std::cerr << "!! cannot open output file " << filename << std::endl;
      exit(0);
    }
    theInstances.push_back(instance);
    return *instance;
  }

  ALIFileOut* instance = new ALIFileOut(filename);  // it will not reach here, only to avoid warning
  return *instance;
}
