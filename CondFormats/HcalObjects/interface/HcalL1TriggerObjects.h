#ifndef HcalL1TriggerObjects_h
#define HcalL1TriggerObjects_h


#include "CondFormats/Serialization/interface/Serializable.h"

#include <cstring>
#include <string>

#include "CondFormats/HcalObjects/interface/HcalL1TriggerObject.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "FWCore/Utilities/interface/Exception.h"

class HcalL1TriggerObjects: public HcalCondObjectContainer<HcalL1TriggerObject>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
 HcalL1TriggerObjects():HcalCondObjectContainer<HcalL1TriggerObject>(nullptr) { }
#endif
  HcalL1TriggerObjects(const HcalTopology* topo):HcalCondObjectContainer<HcalL1TriggerObject>(topo) {}

  //fill the chars and read them
  void setTagString(std::string const& fTag) {
    std::size_t maxCharacters = charArraySize - 1;
    if (fTag.size() > maxCharacters) {
      throw cms::Exception("HcalL1TriggerObjects::setTagString: string exceeds array size");
    }
    strncpy(mTag, fTag.c_str(), maxCharacters);
    mTag[maxCharacters] = '\0';
  }

  void setAlgoString(std::string const& fAlgo) {
    std::size_t maxCharacters = charArraySize - 1;
    if (fAlgo.size() > maxCharacters) {
      throw cms::Exception("HcalL1TriggerObjects::setAlgoString: string exceeds array size");
    }
    strncpy(mAlgo, fAlgo.c_str(), maxCharacters);
    mAlgo[maxCharacters] = '\0';
  }

  std::string getTagString() const { return mTag; }
  std::string getAlgoString() const { return mAlgo; }
  std::string myname() const override { return "HcalL1TriggerObjects"; }

 private:

  static constexpr std::size_t charArraySize = 128;
  char mTag[charArraySize];
  char mAlgo[charArraySize];


 COND_SERIALIZABLE;
};
#endif
