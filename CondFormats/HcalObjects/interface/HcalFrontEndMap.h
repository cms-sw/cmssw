#ifndef HcalFrontEndMap_h
#define HcalFrontEndMap_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <set>
#include <vector>
#include <algorithm>
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalFrontEndId.h"
#include <cstdint>

//forward declaration
namespace HcalFrontEndMapAddons {
  class Helper;
}

class HcalFrontEndMap {
public:
  class PrecisionItem {
  public:
    PrecisionItem() {
      mId = mRM = 0;
      mRBX = "";
    }
    PrecisionItem(uint32_t fId, int fRM, std::string fRBX) : mId(fId), mRM(fRM), mRBX(fRBX) {}
    uint32_t mId;
    int mRM;
    std::string mRBX;

    COND_SERIALIZABLE;
  };

  HcalFrontEndMap() {}
  HcalFrontEndMap(const HcalFrontEndMapAddons::Helper& helper);
  ~HcalFrontEndMap();

  // swap function
  void swap(HcalFrontEndMap& other);
  // copy-ctor
  HcalFrontEndMap(const HcalFrontEndMap& src);
  // copy assignment operator
  HcalFrontEndMap& operator=(const HcalFrontEndMap& rhs);
  // move constructor
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  HcalFrontEndMap(HcalFrontEndMap&& other);
#endif

  /// brief lookup the RM associated with the given logical id
  //return Null item if no such mapping
  const int lookupRM(DetId fId) const;
  const int lookupRMIndex(DetId fId) const;
  const int maxRMIndex() const { return HcalFrontEndId::maxRmIndex; }

  /// brief lookup the RBX associated with the given logical id
  //return Null item if no such mapping
  const std::string lookupRBX(DetId fId) const;
  const int lookupRBXIndex(DetId fId) const;

  std::vector<DetId> allDetIds() const;
  std::vector<int> allRMs() const;
  std::vector<std::string> allRBXs() const;

  const PrecisionItem* findById(uint32_t fId) const;

  // sorting
  void sortById();
  void initialize();

protected:
  std::vector<PrecisionItem> mPItems;
  std::vector<const PrecisionItem*> mPItemsById COND_TRANSIENT;

  COND_SERIALIZABLE;
};

namespace HcalFrontEndMapAddons {
  class LessById {
  public:
    bool operator()(const HcalFrontEndMap::PrecisionItem* a, const HcalFrontEndMap::PrecisionItem* b) const {
      return a->mId < b->mId;
    }
    bool operator()(const HcalFrontEndMap::PrecisionItem& a, const HcalFrontEndMap::PrecisionItem& b) const {
      return a.mId < b.mId;
    }
    bool equal(const HcalFrontEndMap::PrecisionItem* a, const HcalFrontEndMap::PrecisionItem* b) const {
      return a->mId == b->mId;
    }
    bool good(const HcalFrontEndMap::PrecisionItem& a) const { return a.mId; }
  };
  class Helper {
  public:
    Helper();
    /// load a new entry
    bool loadObject(DetId fId, int rm, std::string rbx);

    std::set<HcalFrontEndMap::PrecisionItem, LessById> mPItems;
  };
}  // namespace HcalFrontEndMapAddons

#endif
