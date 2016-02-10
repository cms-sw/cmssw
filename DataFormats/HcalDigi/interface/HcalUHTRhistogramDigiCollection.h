#ifndef DATAFORMATS_HCALDIGI_HCALUHTRHISTOGRAMDIGICOLLECTION_H
#define DATAFORMATS_HCALDIGI_HCALUHTRHISTOGRAMDIGICOLLECTION_H 1

#include <ostream>
#include <vector>
#include <boost/cstdint.hpp>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
class HcalUHTRhistogramDigiCollection;
class HcalUHTRhistogramDigi {
public:
  typedef HcalDetId key_type; ///< For the sorted collection

  HcalUHTRhistogramDigi(int index, HcalUHTRhistogramDigiCollection& collection);
  HcalUHTRhistogramDigi(HcalDetId id, HcalUHTRhistogramDigiCollection& collection);
   
  bool separateCapIds() const;

  bool valid() const;

  const HcalDetId& id() const;
  /// get the contents of the specified bin for the specified capid (0-3)
  uint32_t get(int capid, int bin) const;
  /// get the contents of the specified bin summed over capids
  int getSum(int bin) const;

  /// get the array for the specified capid and channel
  void fillBin(int capid, int bin, uint32_t val);

private:
  HcalUHTRhistogramDigiCollection& theCollection_;
  int index_;

};
//std::ostream& operator<<(std::ostream&, const HcalUHTRhistogramDigi& digi);


class HcalUHTRhistogramDigiCollection {
public:
  const HcalDetId& id(int index);
  int index(HcalDetId id);
  bool separateCapIds() const {return separateCapIds_; }

  const HcalUHTRhistogramDigi digi(int index);
  const HcalUHTRhistogramDigi digi(HcalDetId id);

  int getSum(int bin, int index) const;
  uint32_t get(int capid, int bin, int index) const;
  void fillBin(int capid, int bin, uint32_t val, HcalDetId id);
private:
  std::vector<HcalDetId> ids_;
  std::vector<uint32_t> bins_;
  bool separateCapIds_;
  int binsPerHistogram_;
};

#endif
