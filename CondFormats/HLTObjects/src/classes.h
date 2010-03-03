#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/HLTObjects/interface/HLTPrescaleTable.h"

namespace {
  namespace {
    struct dictionary {
      std::string S;
      unsigned int I;
      std::vector<unsigned int> V;
      std::pair<std::string,std::vector<unsigned int> > PSV;
      std::map<std::string,std::vector<unsigned int> > MSV;
    };
  }
}
