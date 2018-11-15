#ifndef L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboardLUT_h
#define L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboardLUT_h

#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"

#include <vector>
#include <map>

/** labels for ME1a and ME1B */
enum CSCPart {ME1B = 1, ME1A=4, ME21=21, ME1Ag, ME11, ME31, ME41};
enum Parity {Even=0, Odd=1};

class CSCMotherboardLUTME11
{
 public:
  CSCMotherboardLUTME11();
  ~CSCMotherboardLUTME11() {}
  bool doesALCTCrossCLCT(const CSCALCTDigi &a, const CSCCLCTDigi &c,
                         int theEndcap, bool gangedME1a = false) const;
 private:
  // LUT for which ME1/1 wire group can cross which ME1/a halfstrip
  // 1st index: WG number
  // 2nd index: inclusive HS range
  //with "ag" a modified LUT for ganged ME1a
  std::vector<std::vector<double> > lut_wg_vs_hs_me1a;
  std::vector<std::vector<double> > lut_wg_vs_hs_me1ag;
  // LUT for which ME1/1 wire group can cross which ME1/b halfstrip
  // 1st index: WG number
  // 2nd index: inclusive HS range
  std::vector<std::vector<double> > lut_wg_vs_hs_me1b;
};

class CSCGEMMotherboardLUT
{
 public:

  CSCGEMMotherboardLUT();
  virtual ~CSCGEMMotherboardLUT();

  // map of GEM pad number to CSC halfstrip number
  virtual std::vector<int> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const=0;
  // map of CSC halfstrip number to GEM pad number
  virtual std::vector<std::pair<int,int> > get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const=0;
  // map of CSC wiregroup to GEM rols
  std::vector<std::pair<int,int> > get_csc_wg_to_gem_roll(Parity par, int layer=1) const;
  // map the GEM roll to the wire-group in the middle of that roll
  std::vector<int> get_gem_roll_to_csc_wg(Parity par, int layer=1) const;

 protected:

  // maps the edges of the CSC wire group to eta for odd numbered chambers
  std::vector<std::vector<double> > lut_wg_eta_odd;
  // maps the edges of the CSC wire group to eta for even numbered chambers
  std::vector<std::vector<double> > lut_wg_eta_even;
  // LUT with bending angles of the GEM-CSC high efficiency patterns (98%)
  // 1st index: pt value = {3,5,7,10,15,20,30,40}
  // 2nd index: bending angle for odd numbered chambers
  // 3rd index: bending angle for even numbered chambers
  std::vector<std::vector<double> > lut_pt_vs_dphi_gemcsc;
  //LUTs that map CSC trigger geometry to GEM trigger geometry

  // map of roll N to min and max eta
  std::vector<std::pair<double,double> > gem_roll_eta_limits_odd_l1;
  std::vector<std::pair<double,double> > gem_roll_eta_limits_odd_l2;
  std::vector<std::pair<double,double> > gem_roll_eta_limits_even_l1;
  std::vector<std::pair<double,double> > gem_roll_eta_limits_even_l2;

  // map CSC wire-group to GEM roll number
  std::vector<std::pair<int,int> > csc_wg_to_gem_roll_odd_l1;
  std::vector<std::pair<int,int> > csc_wg_to_gem_roll_odd_l2;
  std::vector<std::pair<int,int> > csc_wg_to_gem_roll_even_l1;
  std::vector<std::pair<int,int> > csc_wg_to_gem_roll_even_l2;

  // map the GEM roll to the wire-group in the middle of that roll
  std::vector<int> gem_roll_to_csc_wg_odd_l1;
  std::vector<int> gem_roll_to_csc_wg_odd_l2;
  std::vector<int> gem_roll_to_csc_wg_even_l1;
  std::vector<int> gem_roll_to_csc_wg_even_l2;
};


class CSCGEMMotherboardLUTME11 : public CSCGEMMotherboardLUT
{
 public:

  CSCGEMMotherboardLUTME11();
  ~CSCGEMMotherboardLUTME11() override;

  std::vector<int> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const override;
  std::vector<std::pair<int,int> > get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const override;

  // map of GEM pad to CSC HS for ME1a chambers
  std::vector<int> gem_pad_to_csc_hs_me1a_odd;
  std::vector<int> gem_pad_to_csc_hs_me1a_even;

  // map of GEM pad to CSC HS for ME1b chambers
  std::vector<int> gem_pad_to_csc_hs_me1b_odd;
  std::vector<int> gem_pad_to_csc_hs_me1b_even;

  // map of CSC HS to GEM pad for ME1a chambers
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_me1a_odd;
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_me1a_even;

  // map of CSC HS to GEM pad for ME1b chambers
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_me1b_odd;
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_me1b_even;
};

class CSCGEMMotherboardLUTME21 : public CSCGEMMotherboardLUT
{
 public:

  CSCGEMMotherboardLUTME21();
  ~CSCGEMMotherboardLUTME21() override;

  std::vector<int> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const override;
  std::vector<std::pair<int,int> > get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const override;

  // map of GEM pad to CSC HS
  std::vector<int> gem_pad_to_csc_hs_odd;
  std::vector<int> gem_pad_to_csc_hs_even;

  // map of CSC HS to GEM pad
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_odd;
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_even;
};

#endif
