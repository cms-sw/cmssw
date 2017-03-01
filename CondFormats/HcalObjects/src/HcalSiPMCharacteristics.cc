#include <algorithm>
#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalSiPMCharacteristics::HcalSiPMCharacteristics() : mPItemsByType(nullptr) {}

namespace hcal_impl {
  class LessByType {public: bool operator () (const HcalSiPMCharacteristics::PrecisionItem* a, const HcalSiPMCharacteristics::PrecisionItem* b) {return a->type_ < b->type_;}};
}

HcalSiPMCharacteristics::~HcalSiPMCharacteristics() {
  delete mPItemsByType.load();
}

// copy-ctor
HcalSiPMCharacteristics::HcalSiPMCharacteristics(const HcalSiPMCharacteristics& src)
  : mPItems(src.mPItems), mPItemsByType(nullptr) {}

// copy assignment operator
HcalSiPMCharacteristics& HcalSiPMCharacteristics::operator=(const HcalSiPMCharacteristics& rhs) {
    HcalSiPMCharacteristics temp(rhs);
    temp.swap(*this);
    return *this;
}

// public swap function
void HcalSiPMCharacteristics::swap(HcalSiPMCharacteristics& other) {
  std::swap(mPItems, other.mPItems);
  other.mPItemsByType.exchange(mPItemsByType.exchange(other.mPItemsByType.load(std::memory_order_acquire), 
						      std::memory_order_acq_rel),
			       std::memory_order_acq_rel);
}

// move constructor
HcalSiPMCharacteristics::HcalSiPMCharacteristics(HcalSiPMCharacteristics&& other) : HcalSiPMCharacteristics() {
  other.swap(*this);
}

const HcalSiPMCharacteristics::PrecisionItem* HcalSiPMCharacteristics::findByType (int type) const {
  HcalSiPMCharacteristics::PrecisionItem target(type, 0, 0, 0, 0, 0, 0, 0);
  std::vector<const HcalSiPMCharacteristics::PrecisionItem*>::const_iterator item;

  sortByType();
  auto const& ptr = (*mPItemsByType.load(std::memory_order_acquire));
  item = std::lower_bound (ptr.begin(), ptr.end(), &target, hcal_impl::LessByType());
  if (item == ptr.end() || (*item)->type_ != type)
    //    throw cms::Exception ("Conditions not found") << "Unavailable SiPMCharacteristics for type " << type;
    return 0;
  return *item;
}

bool HcalSiPMCharacteristics::loadObject(int type, int pixels, float parLin1, 
					 float parLin2, float parLin3, 
					 float crossTalk, int auxi1,
					 float auxi2) {
  const HcalSiPMCharacteristics::PrecisionItem* item = findByType(type);
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
    if (mPItemsByType) {
      delete mPItemsByType.load();
      mPItemsByType = nullptr;
    }
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

void HcalSiPMCharacteristics::sortByType () const {
  if (!mPItemsByType.load(std::memory_order_acquire)) {
    auto ptr = new std::vector<const PrecisionItem*>;
    for (auto i=mPItems.begin(); i!=mPItems.end(); ++i) {
      if (i->type_) (*ptr).push_back(&(*i));
    }
    
    std::sort ((*ptr).begin(), (*ptr).end(), hcal_impl::LessByType());
    //atomically try to swap this to become mPItemsByType
    std::vector<const PrecisionItem*>* expect = nullptr;
    bool exchanged = mPItemsByType.compare_exchange_strong(expect, ptr, std::memory_order_acq_rel);
    if(!exchanged) {
      delete ptr;
    }
  }
}
