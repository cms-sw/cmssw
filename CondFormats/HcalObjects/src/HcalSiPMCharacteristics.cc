#include <algorithm>
#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "CondFormats/HcalObjects/interface/HcalObjectAddons.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalSiPMCharacteristics::HcalSiPMCharacteristics(const HcalSiPMCharacteristicsAddons::Helper& helper) :
  mPItems(helper.mPItems.begin(),helper.mPItems.end())
{
  initialize();
}

HcalSiPMCharacteristics::~HcalSiPMCharacteristics() {
}

// copy-ctor
HcalSiPMCharacteristics::HcalSiPMCharacteristics(const HcalSiPMCharacteristics& src)
  : mPItems(src.mPItems), mPItemsByType(src.mPItemsByType) {}

// copy assignment operator
HcalSiPMCharacteristics& HcalSiPMCharacteristics::operator=(const HcalSiPMCharacteristics& rhs) {
    HcalSiPMCharacteristics temp(rhs);
    temp.swap(*this);
    return *this;
}

// public swap function
void HcalSiPMCharacteristics::swap(HcalSiPMCharacteristics& other) {
  std::swap(mPItems, other.mPItems);
  std::swap(mPItemsByType, other.mPItemsByType);
}

// move constructor
HcalSiPMCharacteristics::HcalSiPMCharacteristics(HcalSiPMCharacteristics&& other) : HcalSiPMCharacteristics() {
  other.swap(*this);
}

const HcalSiPMCharacteristics::PrecisionItem* HcalSiPMCharacteristics::findByType (int type) const {

  const HcalSiPMCharacteristics::PrecisionItem* retItem = nullptr;

  for(unsigned int i = 0; i < getTypes(); i++){
    auto iter = &mPItems.at(i);
    if(type==iter->type_) retItem = iter;    
  }
  return retItem;

  //NOT WORKING:
  //PrecisionItem target(type, 0, 0, 0, 0, 0, 0, 0);
  //return HcalObjectAddons::findByT<PrecisionItem,HcalSiPMCharacteristicsAddons::LessByType>(&target,mPItemsByType);
}

HcalSiPMCharacteristicsAddons::Helper::Helper() {}

bool HcalSiPMCharacteristicsAddons::Helper::loadObject(int type, int pixels, float parLin1, 
					 float parLin2, float parLin3, 
					 float crossTalk, int auxi1,
					 float auxi2) {
  HcalSiPMCharacteristics::PrecisionItem target(type,pixels,parLin1, 
						  parLin2,parLin3,crossTalk,
						  auxi1,auxi2);
  auto iter = mPItems.find(target);
  if (iter!=mPItems.end()) {
    edm::LogWarning("HCAL") << "HcalSiPMCharacteristics::loadObject type " 
			    << type << " already exists with pixels "
			    << iter->pixels_ << " NoLinearity parameters " 
			    << iter->parLin1_ << ":" << iter->parLin2_ << ":"
			    << iter->parLin3_ << " CrossTalk parameter "
			    << iter->crossTalk_ << " new values " << pixels
			    << ", " << parLin1 << ", " << parLin2 << ", "
			    << parLin3 << ", " << crossTalk << ", " << auxi1
			    << " and " << auxi2 << " are ignored";
    return false;
  } else {
    mPItems.insert(target);
    return true;
  }
}

int HcalSiPMCharacteristics::getPixels(int type ) const {
  const HcalSiPMCharacteristics::PrecisionItem* item = findByType(type);
  return (item ? item->pixels_ : 0);
}

std::vector<float> HcalSiPMCharacteristics::getNonLinearities(int type) const {
  const HcalSiPMCharacteristics::PrecisionItem* item = findByType(type);
  std::vector<float> pars;
  if (item) {
    pars.push_back(item->parLin1_);
    pars.push_back(item->parLin2_);
    pars.push_back(item->parLin3_);
  }
  return pars;
}

float HcalSiPMCharacteristics::getCrossTalk(int type) const {
  const PrecisionItem* item = findByType(type);
  return (item ? item->crossTalk_ : 0);
}

int HcalSiPMCharacteristics::getAuxi1(int type) const {
  const HcalSiPMCharacteristics::PrecisionItem* item = findByType(type);
  return (item ? item->auxi1_ : 0);
}

float HcalSiPMCharacteristics::getAuxi2(int type) const {
  const HcalSiPMCharacteristics::PrecisionItem* item = findByType(type);
  return (item ? item->auxi2_ : 0);
}

void HcalSiPMCharacteristics::sortByType () {
  HcalObjectAddons::sortByT<PrecisionItem,HcalSiPMCharacteristicsAddons::LessByType>(mPItems,mPItemsByType);
}

void HcalSiPMCharacteristics::initialize(){
  HcalSiPMCharacteristics::sortByType();
}
