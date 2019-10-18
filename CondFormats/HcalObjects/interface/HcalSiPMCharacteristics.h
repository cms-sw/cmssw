#ifndef CondFormatsHcalObjectsHcalSiPMCharacteristics_h
#define CondFormatsHcalObjectsHcalSiPMCharacteristics_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <set>
#include <algorithm>
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif

//forward declarations
namespace HcalSiPMCharacteristicsAddons {
  class Helper;
}

class HcalSiPMCharacteristics {
public:
  class PrecisionItem {
  public:
    PrecisionItem() {
      type_ = auxi1_ = pixels_ = 0;
      parLin1_ = parLin2_ = parLin3_ = crossTalk_ = auxi2_ = 0;
    }
    PrecisionItem(
        int type, int pixels, float parLin1, float parLin2, float parLin3, float crossTalk, int auxi1, float auxi2)
        : type_(type),
          pixels_(pixels),
          parLin1_(parLin1),
          parLin2_(parLin2),
          parLin3_(parLin3),
          crossTalk_(crossTalk),
          auxi1_(auxi1),
          auxi2_(auxi2) {}

    int type_;
    int pixels_;
    float parLin1_;
    float parLin2_;
    float parLin3_;
    float crossTalk_;
    int auxi1_;
    float auxi2_;

    COND_SERIALIZABLE;
  };

  HcalSiPMCharacteristics() {}
  HcalSiPMCharacteristics(const HcalSiPMCharacteristicsAddons::Helper& helper);
  ~HcalSiPMCharacteristics();

  // swap function
  void swap(HcalSiPMCharacteristics& other);
  // copy-ctor
  HcalSiPMCharacteristics(const HcalSiPMCharacteristics& src);
  // copy assignment operator
  HcalSiPMCharacteristics& operator=(const HcalSiPMCharacteristics& rhs);
  // move constructor
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  HcalSiPMCharacteristics(HcalSiPMCharacteristics&& other);
#endif

  /// get # of types
  unsigned int getTypes() const { return mPItems.size(); }
  int getType(unsigned int k) const { return mPItems[k].type_; }
  /// get # of pixels
  int getPixels(int type) const;
  /// get nonlinearity constants
  std::vector<float> getNonLinearities(int type) const;
  /// get cross talk
  float getCrossTalk(int type) const;
  /// get auxiliary words
  int getAuxi1(int type) const;
  float getAuxi2(int type) const;

  const PrecisionItem* findByType(int type) const;

  // sorting
  void sortByType();
  void initialize();

protected:
  std::vector<PrecisionItem> mPItems;
  std::vector<const PrecisionItem*> mPItemsByType COND_TRANSIENT;

  COND_SERIALIZABLE;
};

namespace HcalSiPMCharacteristicsAddons {
  class LessByType {
  public:
    bool operator()(const HcalSiPMCharacteristics::PrecisionItem* a,
                    const HcalSiPMCharacteristics::PrecisionItem* b) const {
      return a->type_ < b->type_;
    }
    bool operator()(const HcalSiPMCharacteristics::PrecisionItem& a,
                    const HcalSiPMCharacteristics::PrecisionItem& b) const {
      return a.type_ < b.type_;
    }
    bool equal(const HcalSiPMCharacteristics::PrecisionItem* a, const HcalSiPMCharacteristics::PrecisionItem* b) const {
      return a->type_ == b->type_;
    }
    bool good(const HcalSiPMCharacteristics::PrecisionItem& a) const { return a.type_; }
  };
  class Helper {
  public:
    Helper();
    // Load a new entry
    bool loadObject(int type,
                    int pixels,
                    float parLin1,
                    float parLin2,
                    float parLin3,
                    float crossTalk,
                    int auxi1 = 0,
                    float auxi2 = 0);

    std::set<HcalSiPMCharacteristics::PrecisionItem, LessByType> mPItems;
  };
}  // namespace HcalSiPMCharacteristicsAddons

#endif
