#ifndef HtrXmlPatternSet_h_included
#define HtrXmlPatternSet_h_included 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "HtrXmlPatternToolParameters.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

class ChannelPattern {
public:
  static const int SAMPLES    = 512;
  static const int NUM_CRATES =  18;
  //this is one larger than it 'needs' to be so that the array index can match the physical slot number
  static const int NUM_SLOTS  =  22;

  ChannelPattern();
  void setLoc(int crate, int slot, int tb, int chan) { m_crate=crate; m_slot=slot; m_tb=tb; m_chan=chan; }
  int getCrate() const { return m_crate; }
  int getSlot() const { return m_slot; }
  int getTB() const { return m_tb; }
  int getChan() const { return m_chan; }
  double& operator[](int bc) { return fCReal[bc]; }
  const double operator[](int bc) const { return fCReal[bc]; }
  void encode();
  int getCoded(int bc) const { return fCCoded[bc]; }
  double getQuantized(int bc) const { return fCQuantized[bc]; }
  void Fill(HtrXmlPatternToolParameters *params,HBHEDigiCollection::const_iterator data);
  void Fill(HtrXmlPatternToolParameters *params,HFDigiCollection::const_iterator data);
  void Fill(HtrXmlPatternToolParameters *params,HODigiCollection::const_iterator data);
  void Fill_by_hand(const HcalElectronicsMap*,int);
private:
  double fCReal[SAMPLES];
  double fCQuantized[SAMPLES];
  int fCCoded[SAMPLES];
  int m_crate, m_slot, m_tb, m_chan;
  int m_sample_pos;
};

class HalfHtrData {
public:
  HalfHtrData(int crate, int slot, int tb);
  ChannelPattern* getPattern(int chan) { return (chan>=1 && chan<=24)?(&m_patterns[chan-1]):(0); }
  int getCrate() const { return m_crate; }
  int getSlot() const { return m_slot; }
  int getTB() const { return m_tb; }
  int getSpigot() const { return m_spigot; }
  int getDCC() const { return m_dcc; }
  void setSpigot(int spigot) { m_spigot = spigot; }
  void setDCC(int dcc) { m_dcc = dcc; }
private:
  ChannelPattern m_patterns[24];
  int m_crate, m_slot, m_tb;
  int m_dcc, m_spigot;
};

class CrateData {
public:
  CrateData(int crate, int slotsActive[ChannelPattern::NUM_SLOTS]);
  ~CrateData();
  HalfHtrData* getHalfHtrData(int slot, int one_two_tb);
private:
  HalfHtrData* m_slotsDoubled[ChannelPattern::NUM_SLOTS][2];
};

class HtrXmlPatternSet {
public:
  HtrXmlPatternSet(int cratesActive[ChannelPattern::NUM_CRATES], int slotsActive[ChannelPattern::NUM_SLOTS]);
  ~HtrXmlPatternSet();
  CrateData* getCrate(int crate);
private:
  CrateData* m_crates[ChannelPattern::NUM_CRATES];
};

#endif
