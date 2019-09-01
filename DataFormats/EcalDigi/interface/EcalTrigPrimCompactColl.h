#ifndef ECALTRIGPRIMCOMPACTCOLL_H
#define ECALTRIGPRIMCOMPACTCOLL_H

#include <vector>
#include <cinttypes>
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/Common/interface/SortedCollection.h"
typedef edm::SortedCollection<EcalTriggerPrimitiveDigi> EcalTrigPrimDigiCollection;

/** \class EcalTrigPrimCompactColl

This collection is used to store ECAL trigger primitive with a low footpring in ROOT EDM file.

Note that only the time sample of interest time is stored. In nornal operation data contains only this time sample.

The interface is similar to EcalTriggerPrimitiveSample one. The collection can also be converted to an EcalTrigPrimDigiCollection
with the method toEcalTrigPrimDigiCollection().

The collection is generated with EcalCompactTrigPrimProducer module from package RecoLocalCalo/EcalRecProducers.

author: Ph. Gras CEA/IRFU Saclay

*/

class EcalTrigPrimCompactColl {
private:
  static const int nPhiBins = 72;
  static const int nEtaBins = 56;
  static const int nBins = nPhiBins * nEtaBins;

private:
  static size_t index(int ieta, int iphi) {
    size_t r = unsigned(((ieta < 0) ? (ieta + 28) : (ieta + 27)) * nPhiBins + (iphi - 1));
    if (r >= (unsigned)nBins)
      throw cms::Exception("Invalid argument")
          << "Trigger tower index (" << ieta << "," << iphi << ") are out of range";
    return r;
  }

public:
  EcalTrigPrimCompactColl() : formatVers_(0), data_(nBins){};

  ///Set data
  void setValue(int ieta, int iphi, uint16_t sample) { data_[index(ieta, iphi)] = sample; }

  //@{
  /// get the raw word
  uint16_t raw(int ieta, int iphi) const { return data_[index(ieta, iphi)]; }
  uint16_t raw(const EcalTrigTowerDetId& ttId) const { return raw(ttId.ieta(), ttId.iphi()); }
  //@}

  //@{
  /// get the encoded/compressed Et (8 bits)
  int compressedEt(int ieta, int iphi) const { return raw(ieta, iphi) & 0xFF; }
  int compressedEt(const EcalTrigTowerDetId& ttId) const { return compressedEt(ttId.ieta(), ttId.iphi()); }
  //@}

  //@{
  /// get the fine-grain bit (1 bit)
  bool fineGrain(int ieta, int iphi) const { return (raw(ieta, iphi) & 0x100) != 0; }
  bool fineGrain(const EcalTrigTowerDetId& ttId) const { return fineGrain(ttId.ieta(), ttId.iphi()); }
  //@}

  //@{
  /// get the Trigger tower Flag (3 bits)
  int ttFlag(int ieta, int iphi) const { return (raw(ieta, iphi) >> 9) & 0x7; }
  int ttFlag(const EcalTrigTowerDetId& ttId) const { return ttFlag(ttId.ieta(), ttId.iphi()); }
  //@}

  //@{
  /// Gets the "strip fine grain veto bit" (sFGVB)  used as L1A spike detection
  /// @return 0 spike like pattern
  ///         1 EM shower like pattern
  int sFGVB(int ieta, int iphi) const { return (raw(ieta, iphi) >> 12) & 0x1; }
  int sFGVB(const EcalTrigTowerDetId& ttId) const { return sFGVB(ttId.ieta(), ttId.iphi()); }
  //@}

  //@{
  /// Gets the "strip fine grain veto bit" (sFGVB)  used as L1A spike detection
  /// Deprecated. Use instead sFGVB() method. Indeed the name of the method being missleading,
  /// since it returns 0 for spike-compatible deposit.
  /// @return 0 spike-like pattern
  ///         1 EM-shower-like pattern
  int l1aSpike(int ieta, int iphi) const { return sFGVB(ieta, iphi); }
  int l1aSpike(const EcalTrigTowerDetId& ttId) const { return sFGVB(ttId); }
  //@}

  void toEcalTrigPrimDigiCollection(EcalTrigPrimDigiCollection& dest) const;

private:
  int16_t formatVers_;
  std::vector<uint16_t> data_;
};

#endif  //ECALTRIGPRIMCOMPACTCOLL_H not defined
