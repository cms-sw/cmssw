#ifndef L1Trigger_CSCTriggerPrimitives_CSCIntegratedMotherboardLUT_h
#define L1Trigger_CSCTriggerPrimitives_CSCIntegratedMotherboardLUT_h

#include <vector>
#include <map>

/** labels for ME1a and ME1B */
enum CSCPart {ME1B = 1, ME1A=4, ME21=21, ME1Ag, ME11, ME31, ME41};

class CSCGEMMotherboardLUT
{
 public:
  
  CSCGEMMotherboardLUT();
  virtual ~CSCGEMMotherboardLUT();
  
  virtual std::map<int,int>* get_gem_pad_to_csc_hs(bool isEven, enum CSCPart) const=0;
  virtual std::map<int,std::pair<int,int>>* get_csc_hs_to_gem_pad(bool isEven, enum CSCPart) const=0;
  std::map<int,std::pair<int,int>>* get_csc_wg_to_gem_roll(bool isEven) const {
    return isEven ? csc_wg_to_gem_roll_odd : csc_wg_to_gem_roll_even;
  }

 protected:

  std::vector<std::vector<double>> * lut_wg_eta_odd;
  std::vector<std::vector<double>> * lut_wg_eta_even;
  // LUT with bending angles of the GEM-CSC high efficiency patterns (98%)
  // 1st index: pt value = {3,5,7,10,15,20,30,40}
  // 2nd index: bending angle for odd numbered chambers
  // 3rd index: bending angle for even numbered chambers
  std::vector<std::vector<double>> * lut_pt_vs_dphi_gemcsc;
  //LUTs that map CSC trigger geometry to GEM trigger geometry

  // map of roll N to min and max eta
  std::map<int,std::pair<double,double> >* gem_roll_eta_limits_odd;
  std::map<int,std::pair<double,double> >* gem_roll_eta_limits_even;

  std::map<int,std::pair<int,int>>* csc_wg_to_gem_roll_odd;
  std::map<int,std::pair<int,int>>* csc_wg_to_gem_roll_even;
};


class CSCGEMMotherboardLUTME11 : public CSCGEMMotherboardLUT
{
 public:

  CSCGEMMotherboardLUTME11();
  virtual ~CSCGEMMotherboardLUTME11();

  virtual std::map<int,int>* get_gem_pad_to_csc_hs(bool isEven, enum CSCPart) const;
  virtual std::map<int,std::pair<int,int>>* get_csc_hs_to_gem_pad(bool isEven, enum CSCPart) const;
  virtual std::vector<std::vector<double>> * get_lut_wg_vs_hs(enum CSCPart) const;

  // LUT for which ME1/1 wire group can cross which ME1/a halfstrip
  // 1st index: WG number
  // 2nd index: inclusive HS range
  //with "ag" a modified LUT for ganged ME1a
  std::vector<std::vector<double>> * lut_wg_vs_hs_me1a;
  std::vector<std::vector<double>> * lut_wg_vs_hs_me1ag;
  std::vector<std::vector<double>> * lut_wg_vs_hs_me1b;

  // map of pad to HS
  std::map<int,int>* gem_pad_to_csc_hs_me1a_odd;
  std::map<int,int>* gem_pad_to_csc_hs_me1a_even;

  std::map<int,int>* gem_pad_to_csc_hs_me1b_odd;
  std::map<int,int>* gem_pad_to_csc_hs_me1b_even;

  std::map<int,std::pair<int,int>>* csc_hs_to_gem_pad_me1a_odd;
  std::map<int,std::pair<int,int>>* csc_hs_to_gem_pad_me1a_even;

  std::map<int,std::pair<int,int>>* csc_hs_to_gem_pad_me1b_odd;
  std::map<int,std::pair<int,int>>* csc_hs_to_gem_pad_me1b_even;
};

class CSCGEMMotherboardLUTME21 : public CSCGEMMotherboardLUT
{
 public:
  
  CSCGEMMotherboardLUTME21();
  virtual ~CSCGEMMotherboardLUTME21();
  
  virtual std::map<int,int>* get_gem_pad_to_csc_hs(bool isEven, enum CSCPart) const;
  virtual std::map<int,std::pair<int,int>>* get_csc_hs_to_gem_pad(bool isEven, enum CSCPart) const;
  
  // map of pad to HS
  std::map<int,int>* gem_pad_to_csc_hs_odd;
  std::map<int,int>* gem_pad_to_csc_hs_even;

  std::map<int,std::pair<int,int>>* csc_hs_to_gem_pad_odd;
  std::map<int,std::pair<int,int>>* csc_hs_to_gem_pad_even;
};

class CSCRPCMotherboardLUT
{
public:

  CSCRPCMotherboardLUT();
  virtual ~CSCRPCMotherboardLUT();

  std::vector<std::vector<double>> * get_lut_wg_eta(bool isEven) const;
  std::map<int,std::pair<double,double> >* get_rpc_roll_eta_limits(bool isEven) const;
  std::map<int,int>* get_rpc_strip_to_csc_hs(bool isEven) const;
  std::map<int,std::pair<int,int>>* get_csc_hs_to_rpc_strip(bool isEven) const;
  std::map<int,int>* get_csc_wg_to_rpc_roll(bool isEven) const;

 protected:
  // map of wg to eta roll
  std::vector<std::vector<double>> * lut_wg_eta_odd;
  std::vector<std::vector<double>> * lut_wg_eta_even;

  // map of roll N to min and max eta
  std::map<int,std::pair<double,double> >* rpc_roll_eta_limits_odd;
  std::map<int,std::pair<double,double> >* rpc_roll_eta_limits_even;

  // map of strip to HS
  std::map<int,int>* rpc_strip_to_csc_hs_odd;
  std::map<int,int>* rpc_strip_to_csc_hs_even;

  std::map<int,std::pair<int,int>>* csc_hs_to_rpc_strip_odd;
  std::map<int,std::pair<int,int>>* csc_hs_to_rpc_strip_even;

  std::map<int,int>* csc_wg_to_rpc_roll_odd;
  std::map<int,int>* csc_wg_to_rpc_roll_even;
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
