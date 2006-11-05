#ifndef L1RCTJetSummaryCard_h
#define L1RCTJetSummaryCard_h

#include <bitset>
#include <vector>
#include <algorithm>

using std::sort;
using std::bitset;
using std::vector;

class L1RCTLookupTables;

class L1RCTJetSummaryCard
{
 public:
  
  //There is no default constructor.
  //It is required to have a crate number attached to it
  //for bookeeping purposes.  
  L1RCTJetSummaryCard(int crtNo);
  
  int crateNumber() {return crtNo;}

  // eGamma Objects
  // The object is defined by a 6 bit rank (temporarily set to 7 bit linear ET)
  // And a position defined by (crdNo and rgnNo)
  // The top four from each crate are returned
  // The order of the candidates is not defined in hardware
  // although, in the case of the emulator they may always be in
  // the descending order of rank

  vector<unsigned short> getIsolatedEGObjects() {return isolatedEGObjects;}
  vector<unsigned short> getNonisolatedEGObjects() {return nonisolatedEGObjects;}
  
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

  vector<unsigned short> getJetRegions() {return jetRegions;}
  vector<unsigned short> getBarrelRegions() {return barrelRegions;}
  vector<unsigned short> getHFRegions() {return HFRegions;}

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

  vector<unsigned short> getHFFineGrainBits() {return hfFineGrainBits;}

  void fillHFRegionSums(vector<unsigned short> hfRegionSums, L1RCTLookupTables *lut);
  void fillRegionSums(vector<unsigned short> regSums){
    barrelRegions = regSums;
  }
  void fillJetRegions();

  void fillIsolatedEGObjects(vector<unsigned short> isoElectrons);
  void fillNonIsolatedEGObjects(vector<unsigned short> nonIsoElectrons);

  void fillMIPBits(vector<unsigned short> mip);
  void fillTauBits(vector<unsigned short> tau);
  void fillOverFlowBits(vector<unsigned short> overflow);
  void fillQuietBits();

  void print();
 private:

  vector<unsigned short> isolatedEGObjects;
  vector<unsigned short> nonisolatedEGObjects;
  vector<unsigned short> jetRegions;

  vector<unsigned short> HFRegions;  // 8-bit et + fine grain?
  vector<unsigned short> barrelRegions;  // no, this is 10-bit et, not (activityBit)(etIn9Bits)(HE_FGBit)(etIn7Bits)

  unsigned short mipBits;
  unsigned short quietBits;
  unsigned short tauBits;
  unsigned short overFlowBits;

  vector<unsigned short> hfFineGrainBits;

  int crtNo;

  unsigned short quietThreshold;

  // Disabled constructors and operators

  L1RCTJetSummaryCard();
};
#endif
