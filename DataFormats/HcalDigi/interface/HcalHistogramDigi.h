#ifndef DATAFORMATS_HCALDIGI_HCALHISTOGRAMDIGI_H
#define DATAFORMATS_HCALDIGI_HCALHISTOGRAMDIGI_H 1

#include <ostream>
#include <boost/cstdint.hpp>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

/** \class HcalHistogramDigi
  *  
  * $Date: 2005/11/14 22:46:50 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalHistogramDigi {
public:
  typedef HcalDetId key_type; ///< For the sorted collection

  HcalHistogramDigi(); // For persistence
  explicit HcalHistogramDigi(const HcalDetId& id);

  const HcalDetId& id() const { return id_; }
  /// get the contents of the specified bin for the specified capid (0-3)
  uint16_t get(int capid, int bin) const;
  /// get the contents of the specified bin summed over capids
  int getSum(int bin) const;

  /// get the array for the specified capid
  uint16_t* getArray(int capid);

  static const int BINS_PER_HISTOGRAM=32;
private:
  HcalDetId id_;
  uint16_t bins_[BINS_PER_HISTOGRAM*4];
};

std::ostream& operator<<(std::ostream&, const HcalHistogramDigi& digi);

#endif
