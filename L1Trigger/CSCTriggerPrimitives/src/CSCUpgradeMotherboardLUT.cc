#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUT.h"

CSCGEMMotherboardLUT::CSCGEMMotherboardLUT() 
  : lut_wg_eta_odd(0)
  , lut_wg_eta_even(0)
  , lut_pt_vs_dphi_gemcsc(0)

  , gem_roll_eta_limits_odd_l1(0)
  , gem_roll_eta_limits_odd_l2(0)
  , gem_roll_eta_limits_even_l1(0)
  , gem_roll_eta_limits_even_l2(0)
    
  , csc_wg_to_gem_roll_odd_l1(0)
  , csc_wg_to_gem_roll_odd_l2(0)
  , csc_wg_to_gem_roll_even_l1(0) 
  , csc_wg_to_gem_roll_even_l2(0)
{
}

std::vector<std::pair<int,int> >
CSCGEMMotherboardLUT::get_csc_wg_to_gem_roll(Parity par, int layer) const
{
  if (par==Parity::Even){ return layer==1 ? csc_wg_to_gem_roll_even_l1 : csc_wg_to_gem_roll_even_l2; }
  else                  { return layer==1 ? csc_wg_to_gem_roll_odd_l1 :  csc_wg_to_gem_roll_odd_l2; }
} 

std::vector<int>
CSCGEMMotherboardLUT::get_gem_roll_to_csc_wg(Parity par, int layer) const
{
  if (par==Parity::Even){ return layer==1 ? gem_roll_to_csc_wg_even_l1 : gem_roll_to_csc_wg_even_l2; }
  else                  { return layer==1 ? gem_roll_to_csc_wg_odd_l1 :  gem_roll_to_csc_wg_odd_l2; }
}

std::vector<int> 
CSCGEMMotherboardLUTME11::get_gem_pad_to_csc_hs(Parity par, enum CSCPart p) const
{
  if (p==CSCPart::ME1A) { return par==Parity::Even ? gem_pad_to_csc_hs_me1a_even : gem_pad_to_csc_hs_me1a_odd; }
  else                  { return par==Parity::Even ? gem_pad_to_csc_hs_me1b_even : gem_pad_to_csc_hs_me1b_odd; }
}

std::vector<int> 
CSCGEMMotherboardLUTME21::get_gem_pad_to_csc_hs(Parity par, enum CSCPart p) const
{
  return par==Parity::Even ? gem_pad_to_csc_hs_even : gem_pad_to_csc_hs_odd;
}

std::vector<std::pair<int,int> > 
CSCGEMMotherboardLUTME21::get_csc_hs_to_gem_pad(Parity par, enum CSCPart p) const
{
  return par==Parity::Even ? csc_hs_to_gem_pad_even : csc_hs_to_gem_pad_odd;
}

std::vector<std::pair<int,int> > 
CSCGEMMotherboardLUTME11::get_csc_hs_to_gem_pad(Parity par, enum CSCPart p) const
{
  if (p==CSCPart::ME1A) { return par==Parity::Even ? csc_hs_to_gem_pad_me1a_even : csc_hs_to_gem_pad_me1a_odd; }
  else                  { return par==Parity::Even ? csc_hs_to_gem_pad_me1b_even : csc_hs_to_gem_pad_me1b_odd; }
}

std::vector<std::vector<double> > 
CSCGEMMotherboardLUTME11::get_lut_wg_vs_hs(enum CSCPart p) const
{
  if (p==CSCPart::ME1A)      { return lut_wg_vs_hs_me1a;  }
  else if (p==CSCPart::ME1B) { return lut_wg_vs_hs_me1b;  }
  else                       { return lut_wg_vs_hs_me1ag; } 
}

CSCGEMMotherboardLUT::~CSCGEMMotherboardLUT()
{
}


CSCGEMMotherboardLUTME11::CSCGEMMotherboardLUTME11() 
  : CSCGEMMotherboardLUT()
{
  lut_wg_eta_odd = {};
  lut_wg_eta_even = {};

  /*
    98% acceptance cuts of the GEM-CSC bending angle in ME1b
    for various pT thresholds and for even/odd chambers
   */
  lut_pt_vs_dphi_gemcsc = {
    {3, 0.03971647, 0.01710244},
    {5, 0.02123785, 0.00928431},
    {7, 0.01475524, 0.00650928},
    {10, 0.01023299, 0.00458796},
    {15, 0.00689220, 0.00331313},
    {20, 0.00535176, 0.00276152},
    {30, 0.00389050, 0.00224959},
    {40, 0.00329539, 0.00204670}};

  lut_wg_vs_hs_me1a  =  {
    {0, 95},{0, 95},{0, 95},{0, 95},{0, 95},
    {0, 95},{0, 95},{0, 95},{0, 95},{0, 95},
    {0, 95},{0, 95},{0, 77},{0, 61},{0, 39},
    {0, 22},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1} };

  lut_wg_vs_hs_me1ag =  {
    {0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
    {0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
    {0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
    {0, 22},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1} };

  lut_wg_vs_hs_me1b  =  {
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
    {100, 127},{73, 127},{47, 127},{22, 127},{0, 127},
    {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
    {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
    {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
    {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
    {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
    {0, 127},{0, 127},{0, 127},{0, 127},{0, 105},
    {0, 93},{0, 78},{0, 63} };
  
  gem_roll_eta_limits_odd_l1 = {};
  gem_roll_eta_limits_odd_l2 = {};
  gem_roll_eta_limits_even_l1 = {};
  gem_roll_eta_limits_even_l2 = {};
  
  csc_wg_to_gem_roll_odd_l1 = {};
  csc_wg_to_gem_roll_odd_l2 = {};
  csc_wg_to_gem_roll_even_l1 = {};
  csc_wg_to_gem_roll_even_l2 = {};
  
  gem_roll_to_csc_wg_odd_l1 = {};
  gem_roll_to_csc_wg_odd_l2 = {};
  gem_roll_to_csc_wg_even_l1 = {};
  gem_roll_to_csc_wg_even_l2 = {};

  gem_pad_to_csc_hs_me1a_odd = {};
  gem_pad_to_csc_hs_me1b_odd = {};
  gem_pad_to_csc_hs_me1a_even = {};
  gem_pad_to_csc_hs_me1b_even = {};
  
  // LUTs for positive endcap differ from negative endcap!
  csc_hs_to_gem_pad_me1a_odd = {};
  csc_hs_to_gem_pad_me1a_even = {};
  csc_hs_to_gem_pad_me1b_odd = {};
  csc_hs_to_gem_pad_me1b_even = {};
}

CSCGEMMotherboardLUTME11::~CSCGEMMotherboardLUTME11() 
{
}


CSCGEMMotherboardLUTME21::CSCGEMMotherboardLUTME21() :   CSCGEMMotherboardLUT()
{
  lut_wg_eta_odd = {};
  lut_wg_eta_even = {};
  
  /*
    98% acceptance cuts of the GEM-CSC bending angle in ME21
    for various pT thresholds and for even/odd chambers
   */
  lut_pt_vs_dphi_gemcsc = {
    {3, 0.01832829, 0.01003643 },
    {5, 0.01095490, 0.00631625 },
    {7, 0.00786026, 0.00501017 },
    {10, 0.00596349, 0.00414560 },
    {15, 0.00462411, 0.00365550 },
    {20, 0.00435298, 0.00361550 },
    {30, 0.00465160, 0.00335700 },
    {40, 0.00372145, 0.00366262 }
  };

  gem_roll_eta_limits_odd_l1 = {};
  gem_roll_eta_limits_odd_l2 = {};
  gem_roll_eta_limits_even_l1 = {};
  gem_roll_eta_limits_even_l2 = {};
  
  csc_wg_to_gem_roll_odd_l1 = {};
  csc_wg_to_gem_roll_even_l1 = {};
  csc_wg_to_gem_roll_odd_l2 = {};
  csc_wg_to_gem_roll_even_l2 = {};

  gem_pad_to_csc_hs_odd = {};
  gem_pad_to_csc_hs_even = {};

  csc_hs_to_gem_pad_odd = {};
  csc_hs_to_gem_pad_even = {};
}


CSCGEMMotherboardLUTME21::~CSCGEMMotherboardLUTME21() 
{
}


CSCRPCMotherboardLUT::CSCRPCMotherboardLUT() 
  : lut_wg_eta_odd(0)
  , lut_wg_eta_even(0)
    
  , rpc_roll_eta_limits_odd(0)
  , rpc_roll_eta_limits_even(0)
    
  , rpc_strip_to_csc_hs_odd(0)
  , rpc_strip_to_csc_hs_even(0)

  , csc_hs_to_rpc_strip_odd(0)
  , csc_hs_to_rpc_strip_even(0)

  , csc_wg_to_rpc_roll_odd(0)
  , csc_wg_to_rpc_roll_even(0)
{
}

CSCRPCMotherboardLUT::~CSCRPCMotherboardLUT()
{
}

CSCRPCMotherboardLUTME31::CSCRPCMotherboardLUTME31() 
  : CSCRPCMotherboardLUT()
{
}

CSCRPCMotherboardLUTME41::CSCRPCMotherboardLUTME41() 
  : CSCRPCMotherboardLUT()
{
}

std::vector<std::vector<double> > 
CSCRPCMotherboardLUT::get_lut_wg_eta(Parity par) const
{
  return par==Parity::Even ? lut_wg_eta_even : lut_wg_eta_odd;
}

std::vector<std::pair<double,double> > 
CSCRPCMotherboardLUT::get_rpc_roll_eta_limits(Parity par) const
{
  return par==Parity::Even ? rpc_roll_eta_limits_even : rpc_roll_eta_limits_odd;
}

std::vector<int> 
CSCRPCMotherboardLUT::get_rpc_strip_to_csc_hs(Parity par) const
{
  return par==Parity::Even ? rpc_strip_to_csc_hs_even : rpc_strip_to_csc_hs_odd;
}

std::vector<std::pair<int,int> > 
CSCRPCMotherboardLUT::get_csc_hs_to_rpc_strip(Parity par) const
{
  return par==Parity::Even ? csc_hs_to_rpc_strip_even : csc_hs_to_rpc_strip_odd;
}

std::vector<int> 
CSCRPCMotherboardLUT::get_csc_wg_to_rpc_roll(Parity par) const
{
  return par==Parity::Even ? csc_wg_to_rpc_roll_even : csc_wg_to_rpc_roll_odd;
}
