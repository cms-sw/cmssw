#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

namespace { 
  struct dictionary {
    std::vector<std::pair<unsigned int,float> > am7;
    std::map<unsigned int,std::vector<std::pair<unsigned int,float> > > am8;
  };
}

