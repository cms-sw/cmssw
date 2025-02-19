#ifndef CASTORTRIGGERPRIMITIVEDIGI_H
#define CASTORTRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

/** \class CastorTriggerPrimitiveDigi
    
$Date: 2010/06/30 12:19:09 $
$Revision: 1.1 $
\author A. Campbell - DESY
*/
class CastorTriggerPrimitiveDigi {
public:
  typedef HcalCastorDetId key_type; ///< For the sorted collection

  CastorTriggerPrimitiveDigi(); // for persistence
  explicit CastorTriggerPrimitiveDigi(const HcalCastorDetId& id);
  
  const HcalCastorDetId& id() const { return id_; }
  int size() const { return (size_&0xF); }
  int presamples() const { return hcalPresamples_&0xF; }

   /// was ZS MarkAndPass?
  bool zsMarkAndPass() const { return (hcalPresamples_&0x10); }
  /// was ZS unsuppressed?
  bool zsUnsuppressed() const { return (hcalPresamples_&0x20); }

  void setZSInfo(bool unsuppressed, bool markAndPass);

  const HcalTriggerPrimitiveSample& operator[](int i) const { return data_[i]; }
  const HcalTriggerPrimitiveSample& sample(int i) const { return data_[i]; }

  /// Full "Sample of Interest"
  const HcalTriggerPrimitiveSample& t0() const { return data_[presamples()]; }  
  /// Fine-grain bit for the "Sample of Interest"
  bool SOI_fineGrain() const { return t0().fineGrain(); }
  /// Compressed ET for the "Sample of Interest"
  int SOI_compressedEt() const { return t0().compressedEt(); }

  int tpchannel(int i) const { return ( ( data_[i].raw() & 0xf800 ) >> 11 ); }
  int tpdata(int i) const { return ( data_[i].raw() & 0x01ff ); }
  bool isSOI(int i ) const { return ( ( data_[i].raw() & 0x0200 ) == 0x0200 ) ; }
  int SOI_tpchannel() const { return ( ( data_[presamples()].raw() & 0xf800 ) >> 11 ); }
  int SOI_tpdata() const { return ( data_[presamples()].raw() & 0x01ff ); }

  void setSize(int size);
  void setPresamples(int ps);
  void setSample(int i, const HcalTriggerPrimitiveSample& sam) { data_[i]=sam; }
  
  static const int MAXSAMPLES = 10;
private:
  HcalCastorDetId id_;
  int size_;
  int hcalPresamples_;
  HcalTriggerPrimitiveSample data_[MAXSAMPLES];
};

std::ostream& operator<<(std::ostream& s, const CastorTriggerPrimitiveDigi& digi);


#endif
