#ifndef OMTF_OMTFPaternMaker_H
#define OMTF_OMTFPaternMaker_H

#include <map>

#include "L1Trigger/L1TMuonOverlap/interface/GoldenPattern.h"

class OMTFConfiguration;
class XMLConfigReader;
class OMTFinput;

namespace edm{
class ParameterSet;
}

class OMTFPaternMaker{

 public:

  OMTFPaternMaker(const edm::ParameterSet & cfg);

  ~OMTFPaternMaker();
  
 private:

  ///Map holding Golden Patterns
  std::map<Key,GoldenPattern*> theGPs;

};


#endif
