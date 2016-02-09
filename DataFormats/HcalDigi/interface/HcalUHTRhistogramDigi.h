#ifndef DATAFORMATS_HCALDIGI_HCALUHTRHISTOGRAMDIGI_H
#define DATAFORMATS_HCALDIGI_HCALUHTRHISTOGRAMDIGI_H 1

#include <ostream>
#include <vector>
#include <boost/cstdint.hpp>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalUHTRhistogramDigi {
public:
  typedef HcalDetId key_type; ///< For the sorted collection

  HcalUHTRhistogramDigi(int nbins = 33, bool sepCapIds = true); // For persistence
  explicit HcalUHTRhistogramDigi(int nbins = 33, bool sepCapIds = true, const HcalDetId& id = 0);

  const HcalDetId& id() const { return id_; }
  const int& nb() const { return nb_; }
  const bool& sc() const { return sc_; }
  /// get the contents of the specified bin for the specified capid (0-3)
  uint32_t get(int capid, int bin) const;
  /// get the contents of the specified bin summed over capids
  int getSum(int bin) const;

  /// get the array for the specified capid and channel
  bool fillBin(int capid, int bin, uint32_t val);

private:
  HcalDetId id_;
  //CapIds<Bins>
  std::vector<std::vector<uint32_t> > histo_;
  bool sc_;
  int nb_;
};

std::ostream& operator<<(std::ostream&, const HcalUHTRhistogramDigi& digi);

#endif
