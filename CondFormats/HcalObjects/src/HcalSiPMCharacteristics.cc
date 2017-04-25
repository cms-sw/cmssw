#include <algorithm>
#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace hcal_impl {
  class LessByType {public: bool operator () (const HcalSiPMCharacteristics::PrecisionItem* a, const HcalSiPMCharacteristics::PrecisionItem* b) {return a->type_ < b->type_;}};
}

HcalSiPMCharacteristics::HcalSiPMCharacteristics(const HcalSiPMCharacteristics::Helper& helper) :
  mPItems(helper.mPItems)
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

const HcalSiPMCharacteristics::PrecisionItem* HcalSiPMCharacteristics::findByType (int type, const std::vector<const PrecisionItem*>& itemsByType) {
  HcalSiPMCharacteristics::PrecisionItem target(type, 0, 0, 0, 0, 0, 0, 0);

  auto item = std::lower_bound (itemsByType.begin(), itemsByType.end(), &target, hcal_impl::LessByType());
  if (item == itemsByType.end() || (*item)->type_ != type)
    //    throw cms::Exception ("Conditions not found") << "Unavailable SiPMCharacteristics for type " << type;
    return 0;
  return *item;
}

HcalSiPMCharacteristics::Helper::Helper() {}

bool HcalSiPMCharacteristics::Helper::loadObject(int type, int pixels, float parLin1, 
					 float parLin2, float parLin3, 
					 float crossTalk, int auxi1,
					 float auxi2) {
  const HcalSiPMCharacteristics::PrecisionItem* item = HcalSiPMCharacteristics::findByType(type,mPItemsByType);
  if (item) {
    edm::LogWarning("HCAL") << "HcalSiPMCharacteristics::loadObject type " 
			    << type << " already exists with pixels "
			    << item->pixels_ << " NoLinearity parameters " 
			    << item->parLin1_ << ":" << item->parLin2_ << ":"
			    << item->parLin3_ << " CrossTalk parameter "
			    << item->crossTalk_ << " new values " << pixels
			    << ", " << parLin1 << ", " << parLin2 << ", "
			    << parLin3 << ", " << crossTalk << ", " << auxi1
			    << " and " << auxi2 << " are ignored";
    return false;
  } else {
    HcalSiPMCharacteristics::PrecisionItem target(type,pixels,parLin1, 
						  parLin2,parLin3,crossTalk,
						  auxi1,auxi2);
    mPItems.push_back(target);
    HcalSiPMCharacteristics::sortByType(mPItems,mPItemsByType);
    return true;
  }
}

int HcalSiPMCharacteristics::getPixels(int type ) const {
  const HcalSiPMCharacteristics::PrecisionItem* item = HcalSiPMCharacteristics::findByType(type,mPItemsByType);
  return (item ? item->pixels_ : 0);
}

std::vector<float> HcalSiPMCharacteristics::getNonLinearities(int type) const {
  const HcalSiPMCharacteristics::PrecisionItem* item = HcalSiPMCharacteristics::findByType(type,mPItemsByType);
  std::vector<float> pars;
  if (item) {
    pars.push_back(item->parLin1_);
    pars.push_back(item->parLin2_);
    pars.push_back(item->parLin3_);
  }
  return pars;
}

float HcalSiPMCharacteristics::getCrossTalk(int type) const {
  const PrecisionItem* item = HcalSiPMCharacteristics::findByType(type,mPItemsByType);
  return (item ? item->crossTalk_ : 0);
}

int HcalSiPMCharacteristics::getAuxi1(int type) const {
  const HcalSiPMCharacteristics::PrecisionItem* item = HcalSiPMCharacteristics::findByType(type,mPItemsByType);
  return (item ? item->auxi1_ : 0);
}

float HcalSiPMCharacteristics::getAuxi2(int type) const {
  const HcalSiPMCharacteristics::PrecisionItem* item = HcalSiPMCharacteristics::findByType(type,mPItemsByType);
  return (item ? item->auxi2_ : 0);
}

void HcalSiPMCharacteristics::sortByType (const std::vector<PrecisionItem>& items, std::vector<const PrecisionItem*>& itemsByType) {
  itemsByType.clear();
  itemsByType.reserve(items.size());
  for(const auto& i : items){
    if (i.type_) itemsByType.push_back(&i);
  }
  std::sort (itemsByType.begin(), itemsByType.end(), hcal_impl::LessByType());
}

void HcalSiPMCharacteristics::initialize(){
  HcalSiPMCharacteristics::sortByType(mPItems,mPItemsByType);
}
