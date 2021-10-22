#ifndef HcalMaterials_h
#define HcalMaterials_h

#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
// place to implement a real working class for material corrections

class HcalMaterial {
public:
  float getValue(float Energy) { return 1.; }
  //  void putValue (unsigned long fId, std::pair<std::vector <float>, std::vector <float> > fArray);

  HcalMaterial(unsigned long fId, const std::pair<std::vector<float>, std::vector<float> >& fCorrs)  //:
  //   mId (fId),
  //   mCorrs (fCorrs)
  {
    mId = fId;
    mCorrs = fCorrs;
  }

  unsigned long mmId(void) { return mId; }

private:
  unsigned long mId;
  std::pair<std::vector<float>, std::vector<float> > mCorrs;
};

class HcalMaterials {
public:
  HcalMaterials();
  ~HcalMaterials();

  float getValue(DetId fId, float energy);
  void putValue(DetId fId, const std::pair<std::vector<float>, std::vector<float> >& fArray);

  typedef HcalMaterial Item;
  typedef std::vector<Item> Container;

private:
  Container mItems;
};

#endif
