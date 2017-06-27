#ifndef L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboardLUT_h
#define L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboardLUT_h

#include <vector>
#include <map>

/** labels for ME1a and ME1B */
enum CSCPart {ME1B = 1, ME1A=4, ME21=21, ME1Ag, ME11, ME31, ME41};
enum Parity {Even=0, Odd=1};

class CSCGEMMotherboardLUT
{
 public:
  
  CSCGEMMotherboardLUT();
  virtual ~CSCGEMMotherboardLUT();
  
  virtual std::vector<int> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const=0;
  virtual std::vector<std::pair<int,int> > get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const=0;
  std::vector<std::pair<int,int> > get_csc_wg_to_gem_roll(Parity par, int layer=1) const;
  // map the GEM roll to the wire-group in the middle of that roll
  std::vector<int> get_gem_roll_to_csc_wg(Parity par, int layer=1) const;

 protected:

  std::vector<std::vector<double> > lut_wg_eta_odd;
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
  virtual ~CSCGEMMotherboardLUTME11();

  virtual std::vector<int> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const;
  virtual std::vector<std::pair<int,int> > get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const;
  virtual std::vector<std::vector<double> > get_lut_wg_vs_hs(enum CSCPart) const;

  // LUT for which ME1/1 wire group can cross which ME1/a halfstrip
  // 1st index: WG number
  // 2nd index: inclusive HS range
  //with "ag" a modified LUT for ganged ME1a
  std::vector<std::vector<double> > lut_wg_vs_hs_me1a;
  std::vector<std::vector<double> > lut_wg_vs_hs_me1ag;
  std::vector<std::vector<double> > lut_wg_vs_hs_me1b;

  // map of pad to HS
  std::vector<int> gem_pad_to_csc_hs_me1a_odd;
  std::vector<int> gem_pad_to_csc_hs_me1a_even;

  std::vector<int> gem_pad_to_csc_hs_me1b_odd;
  std::vector<int> gem_pad_to_csc_hs_me1b_even;

  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_me1a_odd;
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_me1a_even;

  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_me1b_odd;
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_me1b_even;
};

class CSCGEMMotherboardLUTME21 : public CSCGEMMotherboardLUT
{
 public:
  
  CSCGEMMotherboardLUTME21();
  virtual ~CSCGEMMotherboardLUTME21();
  
  virtual std::vector<int> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const;
  virtual std::vector<std::pair<int,int> > get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const;
  
  // map of pad to HS
  std::vector<int> gem_pad_to_csc_hs_odd;
  std::vector<int> gem_pad_to_csc_hs_even;

  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_odd;
  std::vector<std::pair<int,int> > csc_hs_to_gem_pad_even;
};

class CSCRPCMotherboardLUT
{
public:

  CSCRPCMotherboardLUT();
  virtual ~CSCRPCMotherboardLUT();

  std::vector<std::vector<double> > get_lut_wg_eta(Parity par) const;
  std::vector<std::pair<double,double> > get_rpc_roll_eta_limits(Parity par) const;
  std::vector<int> get_rpc_strip_to_csc_hs(Parity par) const;
  std::vector<std::pair<int,int> > get_csc_hs_to_rpc_strip(Parity par) const;
  std::vector<int> get_csc_wg_to_rpc_roll(Parity par) const;

 protected:
  // map of wg to eta roll
  std::vector<std::vector<double> > lut_wg_eta_odd;
  std::vector<std::vector<double> > lut_wg_eta_even;

  // map of roll N to min and max eta
  std::vector<std::pair<double,double> > rpc_roll_eta_limits_odd;
  std::vector<std::pair<double,double> > rpc_roll_eta_limits_even;

  // map of strip to HS
  std::vector<int> rpc_strip_to_csc_hs_odd;
  std::vector<int> rpc_strip_to_csc_hs_even;

  std::vector<std::pair<int,int> > csc_hs_to_rpc_strip_odd;
  std::vector<std::pair<int,int> > csc_hs_to_rpc_strip_even;

  std::vector<int> csc_wg_to_rpc_roll_odd;
  std::vector<int> csc_wg_to_rpc_roll_even;
};

class CSCRPCMotherboardLUTME31 : public CSCRPCMotherboardLUT 
{
 public:
  
  CSCRPCMotherboardLUTME31();
  virtual ~CSCRPCMotherboardLUTME31() {}
};

class CSCRPCMotherboardLUTME41 : public CSCRPCMotherboardLUT 
{
 public:
  
  CSCRPCMotherboardLUTME41();
  virtual ~CSCRPCMotherboardLUTME41() {}  
};

#endif
