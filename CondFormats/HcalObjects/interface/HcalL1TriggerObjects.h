#ifndef HcalL1TriggerObjects_h
#define HcalL1TriggerObjects_h


#include <cstring>
#include <string>

#include "CondFormats/HcalObjects/interface/HcalL1TriggerObject.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalL1TriggerObjects: public HcalCondObjectContainer<HcalL1TriggerObject>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
 HcalL1TriggerObjects():HcalCondObjectContainer<HcalL1TriggerObject>(0) { }
#endif
  HcalL1TriggerObjects(const HcalTopology* topo):HcalCondObjectContainer<HcalL1TriggerObject>(topo) {}

  //fill the chars and read them
  void setTagString(std::string fTag) {strncpy(mTag,fTag.c_str(),128);}
  void setAlgoString(std::string fAlgo) {strncpy(mAlgo,fAlgo.c_str(),128);}

  std::string getTagString() const {return (std::string)mTag;}
  std::string getAlgoString() const {return (std::string)mAlgo;}

  std::string myname() const {return (std::string)"HcalL1TriggerObjects";}

 private:
  char mTag[128];
  char mAlgo[128];

};
#endif
