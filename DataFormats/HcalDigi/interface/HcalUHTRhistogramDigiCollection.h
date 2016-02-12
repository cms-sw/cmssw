#ifndef DATAFORMATS_HCALDIGI_HCALUHTRHISTOGRAMDIGICOLLECTION_H
#define DATAFORMATS_HCALDIGI_HCALUHTRHISTOGRAMDIGICOLLECTION_H 1

#include <ostream>
#include <vector>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
class HcalUHTRhistogramDigiCollection;
class HcalUHTRhistogramDigi {
public:
  HcalUHTRhistogramDigi(size_t index, const HcalUHTRhistogramDigiCollection& collection);
   
  bool separateCapIds() const;

  bool valid() const;

  const HcalDetId& id() const;
  /// get the contents of the specified bin for the specified capid (0-3)
  uint32_t get(int capid, int bin) const;
  /// get the contents of the specified bin summed over capids
  int getSum(int bin) const;



  void fillBin();

private:
  const HcalUHTRhistogramDigiCollection& theCollection_;
protected:
  size_t index_;

};

class HcalUHTRhistogramDigiMutable :public HcalUHTRhistogramDigi {
  private:
    HcalUHTRhistogramDigiCollection& theCollectionMutable_;
  public:
    void fillBin(int capid, int bin, uint32_t val);
    HcalUHTRhistogramDigiMutable(size_t index, HcalUHTRhistogramDigiCollection& collection);
};

class HcalUHTRhistogramDigiCollection {
public:
  HcalUHTRhistogramDigiCollection();
  HcalUHTRhistogramDigiCollection(int numBins, bool sepCapIds);
  static const size_t INVALID = (size_t)-1;
  const size_t find(HcalDetId id) const;
  bool separateCapIds() const { return separateCapIds_; }

  const HcalUHTRhistogramDigi at(size_t index) const;
  const HcalUHTRhistogramDigi operator[](size_t index) const;
  HcalUHTRhistogramDigiMutable addHistogram(const HcalDetId& id);

  const size_t size() const { return ids_.size(); }
  const int binsPerHisto() const { return binsPerHistogram_; }
protected:
  friend class HcalUHTRhistogramDigi; 
  friend class HcalUHTRhistogramDigiMutable;
  const int getSum(int bin, size_t index) const;
  const uint32_t get(int capid, int bin, size_t index) const;
  void fillBin(int capid, int bin, uint32_t val, size_t index);
  const HcalDetId& id(size_t index) const;
private:
  std::vector<HcalDetId> ids_;
  std::vector<uint32_t> bins_;
  bool separateCapIds_;
  int binsPerHistogram_;
};

#endif
