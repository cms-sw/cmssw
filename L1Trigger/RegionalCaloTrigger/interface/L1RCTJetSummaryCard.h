#ifndef L1RCTJetSummaryCard_h
#define L1RCTJetSummaryCard_h

#include <vector>

class L1RCTLookupTables;

class L1RCTJetSummaryCard
{
 public:
  
  //There is no default constructor.
  //It is required to have a crate number attached to it
  //for bookeeping purposes.  
  L1RCTJetSummaryCard(int crtNo, const L1RCTLookupTables* rctLookupTables);
  
  int crateNumber() {return crtNo;}

  // eGamma Objects
  // The object is defined by a 6 bit rank (temporarily set to 7 bit linear ET)
  // And a position defined by (crdNo and rgnNo)
  // The top four from each crate are returned
  // The order of the candidates is not defined in hardware
  // although, in the case of the emulator they may always be in
  // the descending order of rank

  std::vector<unsigned short> getIsolatedEGObjects() {return isolatedEGObjects;}
  std::vector<unsigned short> getNonisolatedEGObjects() {return nonisolatedEGObjects;}
  
  // Region sums 10-bit energy (bits 0-9),overflow (bit 10)
  // bit 11 is tau bit

  // Except for the HF regions, the data is only 8-bit wide
  // packed non-linearly with no interpretation of bits by RCT
  // However, temporarily, we set this to be 10-bit ET as well
  // in bits 0-9

  // The following Jet Summary Card output data are packed
  // in unsigned 16 bit words although in reality they
  // are limited to lower number of bits

  // There are 22 total regions (2 phi x 11 eta) including
  // HB, HE and HF

  // The data are arranged in the vector such that first
  // 11 unsigned values are for lower phi and the next
  // 11 unsigned values are for higher phi in the crate

  // Further, the order of placement in the vector is such
  // that the eta decreases from -5 to 0 for crates 0-8
  // and increases from 0 to +5 for crates 9-17

  std::vector<unsigned short> getJetRegions() {return jetRegions;}
  std::vector<unsigned short> getBarrelRegions() {return barrelRegions;}
  std::vector<unsigned short> getHFRegions() {return HFRegions;}

  // Muon bits consist of 14 quiet bits and 14 MIP bits
  // These are packed into one unsigned short each
  // The labeling of the bits is in the order
  // (crdNo0, rgnNo0), (crdNo0, rgnNo1)
  // (crdNo1, rgnNo0), (crdNo1, rgnNo1)
  // ...
  // (crdNo6, rgnNo0), (crdNo6, rgnNo1)
  // The same ordering is true for Quiet bits also

  unsigned short getMIPBits() {return mipBits;}
  unsigned short getQuietBits() {return quietBits;}

  unsigned short getTauBits() {return tauBits;}
  unsigned short getOverFlowBits() {return overFlowBits;}

  std::vector<unsigned short> getHFFineGrainBits() {return hfFineGrainBits;}

  void fillHFRegionSums(const std::vector<unsigned short>& hfRegionSums);
  void fillRegionSums(const std::vector<unsigned short>& regSums){
    barrelRegions = regSums;
  }
  void fillJetRegions();

  void fillIsolatedEGObjects(const std::vector<unsigned short>& isoElectrons);
  void fillNonIsolatedEGObjects(const std::vector<unsigned short>& nonIsoElectrons);

  void fillMIPBits(const std::vector<unsigned short>& mip);
  void fillTauBits(const std::vector<unsigned short>& tau);
  void fillOverFlowBits(const std::vector<unsigned short>& overflow);
  void fillQuietBits();

  void print();
 private:

  int crtNo;

  const L1RCTLookupTables* rctLookupTables_;

  std::vector<unsigned short> isolatedEGObjects;
  std::vector<unsigned short> nonisolatedEGObjects;
  std::vector<unsigned short> jetRegions;

  std::vector<unsigned short> HFRegions;  // 8-bit et + fine grain?
  std::vector<unsigned short> barrelRegions;  // no, this is 10-bit et, not (activityBit)(etIn9Bits)(HE_FGBit)(etIn7Bits)

  unsigned short mipBits;
  unsigned short quietBits;
  unsigned short tauBits;
  unsigned short overFlowBits;

  std::vector<unsigned short> hfFineGrainBits;

  //unsigned quietThreshold;
  unsigned quietThresholdBarrel;
  unsigned quietThresholdEndcap;

  void asicSort(std::vector<unsigned short>& electrons);
  void asicCompare(std::vector<unsigned short>& array);

  // Disabled constructors and operators

  L1RCTJetSummaryCard();
};
#endif
