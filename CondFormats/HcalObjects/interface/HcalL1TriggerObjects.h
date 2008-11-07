#ifndef HcalL1TriggerObjects_h
#define HcalL1TriggerObjects_h


#include <cstring>
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObject.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"


class HcalL1TriggerObjects: public HcalCondObjectContainer<HcalL1TriggerObject>
{
 public:
  HcalL1TriggerObjects():HcalCondObjectContainer<HcalL1TriggerObject>() {}

  //fill the chars and read them
  void setTagString(char* fTag) {strncpy(mTag,fTag);}
  void setAlgoString(char* fAlgo) {strncpy(mAlgo,fAlgo);}

  char* getTagString() {return mTag;}
  char* getAlgoString() {return mAlgo;}

 private:
  char mTag[128];
  char mAlgo[128];

};
#endif
