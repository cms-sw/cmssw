#ifndef ECALTRIGPRIMCOMPACTCOLL_H
#define ECALTRIGPRIMCOMPACTCOLL_H

#include <vector>
#include <inttypes.h>
#include "FWCore/Utilities/interface/Exception.h"

/** \class EcalTrigPrimCompactColl

This collection is used to store ECAL trigger primitive with a low footpring in ROOT EDM file.

author: Ph. Gras CEA/IRFU Saclay

*/

class EcalTrigPrimCompactColl {

private:
  static const int nPhiBins = 72;
  static const int nEtaBins = 56;
  static const int nBins = nPhiBins*nEtaBins;

private:
  static size_t index(int ieta, int iphi){
    size_t r = unsigned(((ieta<0) ? (ieta+28) : (ieta+27))*nPhiBins + (iphi -1));
    if(r >= (unsigned)nBins) throw cms::Exception("Invalid argument") << "Trigger tower index (" << ieta << "," << iphi << ") are out of range";
    return r;
  }
  
public:
  EcalTrigPrimCompactColl(): formatVers_(0), data_(nBins){};
  
  ///Set data
  void setValue(int ieta, int iphi, uint16_t sample){ data_[index(ieta, iphi)] = sample;}
  /// get the raw word
  uint16_t raw(int ieta, int iphi) const { return data_[index(ieta, iphi)]; }
  /// get the encoded/compressed Et (8 bits)
  int compressedEt(int ieta, int iphi) const { return raw(ieta, iphi)&0xFF; }
  /// get the fine-grain bit (1 bit) 
  bool fineGrain(int ieta, int iphi) const { return (raw(ieta, iphi)&0x100)!=0; }
  /// get the Trigger tower Flag (3 bits)
  int ttFlag(int ieta, int iphi) const { return (raw(ieta, iphi)>>9)&0x7; }

  /// gets the L1A spike detection flag. 
  /// @return 1 if the trigger primitive was forced to zero because a spike was detected by L1 trigger,
  ///         0 otherwise
  int l1aSpike(int ieta, int iphi) const { return (raw(ieta, iphi) >>12) & 0x1; }
  
 private:
  int16_t formatVers_;
  std::vector<uint16_t> data_;
};

#endif //ECALTRIGPRIMCOMPACTCOLL_H not defined
