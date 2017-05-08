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
  
  typedef std::vector<int> vi;
  typedef std::vector<std::vector<double> > vvd;
  typedef std::vector<std::pair<double,double> > vpdd;
  typedef std::vector<std::pair<int,int> > vpii;

  virtual std::unique_ptr<vi> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const=0;
  virtual std::unique_ptr<vpii> get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const=0;
  std::unique_ptr<vpii> get_csc_wg_to_gem_roll(Parity par, int layer=1) const;

 protected:

  std::unique_ptr<vvd> lut_wg_eta_odd;
  std::unique_ptr<vvd> lut_wg_eta_even;
  // LUT with bending angles of the GEM-CSC high efficiency patterns (98%)
  // 1st index: pt value = {3,5,7,10,15,20,30,40}
  // 2nd index: bending angle for odd numbered chambers
  // 3rd index: bending angle for even numbered chambers
  std::unique_ptr<vvd> lut_pt_vs_dphi_gemcsc;
  //LUTs that map CSC trigger geometry to GEM trigger geometry

  // map of roll N to min and max eta
  std::unique_ptr<vpdd> gem_roll_eta_limits_odd_l1;
  std::unique_ptr<vpdd> gem_roll_eta_limits_odd_l2;
  std::unique_ptr<vpdd> gem_roll_eta_limits_even_l1;
  std::unique_ptr<vpdd> gem_roll_eta_limits_even_l2;

  std::unique_ptr<vpii> csc_wg_to_gem_roll_odd_l1;
  std::unique_ptr<vpii> csc_wg_to_gem_roll_odd_l2;
  std::unique_ptr<vpii> csc_wg_to_gem_roll_even_l1;
  std::unique_ptr<vpii> csc_wg_to_gem_roll_even_l2;
};


class CSCGEMMotherboardLUTME11 : public CSCGEMMotherboardLUT
{
 public:

  CSCGEMMotherboardLUTME11();
  virtual ~CSCGEMMotherboardLUTME11();

  virtual std::unique_ptr<vi> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const;
  virtual std::unique_ptr<vpii> get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const;
  virtual std::unique_ptr<vvd> get_lut_wg_vs_hs(enum CSCPart) const;

  // LUT for which ME1/1 wire group can cross which ME1/a halfstrip
  // 1st index: WG number
  // 2nd index: inclusive HS range
  //with "ag" a modified LUT for ganged ME1a
  std::unique_ptr<vvd> lut_wg_vs_hs_me1a;
  std::unique_ptr<vvd> lut_wg_vs_hs_me1ag;
  std::unique_ptr<vvd> lut_wg_vs_hs_me1b;

  // map of pad to HS
  std::unique_ptr<vi> gem_pad_to_csc_hs_me1a_odd;
  std::unique_ptr<vi> gem_pad_to_csc_hs_me1a_even;

  std::unique_ptr<vi> gem_pad_to_csc_hs_me1b_odd;
  std::unique_ptr<vi> gem_pad_to_csc_hs_me1b_even;

  std::unique_ptr<vpii> csc_hs_to_gem_pad_me1a_odd;
  std::unique_ptr<vpii> csc_hs_to_gem_pad_me1a_even;

  std::unique_ptr<vpii> csc_hs_to_gem_pad_me1b_odd;
  std::unique_ptr<vpii> csc_hs_to_gem_pad_me1b_even;
};

class CSCGEMMotherboardLUTME21 : public CSCGEMMotherboardLUT
{
 public:
  
  CSCGEMMotherboardLUTME21();
  virtual ~CSCGEMMotherboardLUTME21();
  
  virtual std::unique_ptr<vi> get_gem_pad_to_csc_hs(Parity par, enum CSCPart) const;
  virtual std::unique_ptr<vpii> get_csc_hs_to_gem_pad(Parity par, enum CSCPart) const;
  
  // map of pad to HS
  std::unique_ptr<vi> gem_pad_to_csc_hs_odd;
  std::unique_ptr<vi> gem_pad_to_csc_hs_even;

  std::unique_ptr<vpii> csc_hs_to_gem_pad_odd;
  std::unique_ptr<vpii> csc_hs_to_gem_pad_even;
};

class CSCRPCMotherboardLUT
{
public:

  CSCRPCMotherboardLUT();
  virtual ~CSCRPCMotherboardLUT();

  std::unique_ptr<vvd> get_lut_wg_eta(Parity par) const;
  std::unique_ptr<vpdd> get_rpc_roll_eta_limits(Parity par) const;
  std::unique_ptr<vi> get_rpc_strip_to_csc_hs(Parity par) const;
  std::unique_ptr<vpii> get_csc_hs_to_rpc_strip(Parity par) const;
  std::unique_ptr<vi> get_csc_wg_to_rpc_roll(Parity par) const;

 protected:
  // map of wg to eta roll
  std::unique_ptr<vvd> lut_wg_eta_odd;
  std::unique_ptr<vvd> lut_wg_eta_even;

  // map of roll N to min and max eta
  std::unique_ptr<vpdd> rpc_roll_eta_limits_odd;
  std::unique_ptr<vpdd> rpc_roll_eta_limits_even;

  // map of strip to HS
  std::unique_ptr<vi> rpc_strip_to_csc_hs_odd;
  std::unique_ptr<vi> rpc_strip_to_csc_hs_even;

  std::unique_ptr<vpii> csc_hs_to_rpc_strip_odd;
  std::unique_ptr<vpii> csc_hs_to_rpc_strip_even;

  std::unique_ptr<vi> csc_wg_to_rpc_roll_odd;
  std::unique_ptr<vi> csc_wg_to_rpc_roll_even;
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
