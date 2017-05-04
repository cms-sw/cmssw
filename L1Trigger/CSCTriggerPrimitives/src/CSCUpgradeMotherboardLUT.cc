#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUT.h"

CSCGEMMotherboardLUT::CSCGEMMotherboardLUT() 
  : lut_wg_eta_odd(0)
  , lut_wg_eta_even(0)
  , lut_pt_vs_dphi_gemcsc(0)
{
}

std::vector<int>* CSCGEMMotherboardLUTME11::get_gem_pad_to_csc_hs(bool isEven, enum CSCPart p) const
{
  if (p==CSCPart::ME1A) { return isEven ? gem_pad_to_csc_hs_me1a_even : gem_pad_to_csc_hs_me1a_odd; }
  else                  { return isEven ? gem_pad_to_csc_hs_me1b_even : gem_pad_to_csc_hs_me1b_odd; }
}

std::vector<int>* CSCGEMMotherboardLUTME21::get_gem_pad_to_csc_hs(bool isEven, enum CSCPart p) const
{
  return isEven ? gem_pad_to_csc_hs_even : gem_pad_to_csc_hs_odd;
}

std::vector<std::pair<int,int> >* CSCGEMMotherboardLUTME21::get_csc_hs_to_gem_pad(bool isEven, enum CSCPart p) const
{
  return isEven ? csc_hs_to_gem_pad_even : csc_hs_to_gem_pad_odd;
}

std::vector<std::pair<int,int> >* CSCGEMMotherboardLUTME11::get_csc_hs_to_gem_pad(bool isEven, enum CSCPart p) const
{
  if (p==CSCPart::ME1A) { return isEven ? csc_hs_to_gem_pad_me1a_even : csc_hs_to_gem_pad_me1a_odd; }
  else                  { return isEven ? csc_hs_to_gem_pad_me1b_even : csc_hs_to_gem_pad_me1b_odd; }
}

std::vector<std::vector<double>> * CSCGEMMotherboardLUTME11::get_lut_wg_vs_hs(enum CSCPart p) const
{
  if (p==CSCPart::ME1A)      { return lut_wg_vs_hs_me1a;  }
  else if (p==CSCPart::ME1B) { return lut_wg_vs_hs_me1b;  }
  else                       { return lut_wg_vs_hs_me1ag; } 
}

CSCGEMMotherboardLUT::~CSCGEMMotherboardLUT(){
  if(lut_wg_eta_odd       ) delete lut_wg_eta_odd       ;
  if(lut_wg_eta_even      ) delete lut_wg_eta_even      ;
  if(lut_pt_vs_dphi_gemcsc) delete lut_pt_vs_dphi_gemcsc;

  if(gem_roll_eta_limits_odd_l1) delete gem_roll_eta_limits_odd_l1;
  if(gem_roll_eta_limits_odd_l2) delete gem_roll_eta_limits_odd_l2;
  if(gem_roll_eta_limits_even_l1) delete gem_roll_eta_limits_even_l1;
  if(gem_roll_eta_limits_even_l2) delete gem_roll_eta_limits_even_l2;
}


CSCGEMMotherboardLUTME11::CSCGEMMotherboardLUTME11() 
  : CSCGEMMotherboardLUT()
{
  lut_wg_eta_odd = new std::vector<std::vector<double>> {
  };
  lut_wg_eta_even = new std::vector<std::vector<double>> {
  };


  lut_pt_vs_dphi_gemcsc = new std::vector<std::vector<double>> {
    {3, 0.03971647, 0.01710244},
    {5, 0.02123785, 0.00928431},
    {7, 0.01475524, 0.00650928},
    {10, 0.01023299, 0.00458796},
    {15, 0.00689220, 0.00331313},
    {20, 0.00535176, 0.00276152},
    {30, 0.00389050, 0.00224959},
    {40, 0.00329539, 0.00204670}};

  lut_wg_vs_hs_me1a  = new std::vector<std::vector<double>> {
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

  lut_wg_vs_hs_me1ag = new std::vector<std::vector<double>> {
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

  lut_wg_vs_hs_me1b  = new std::vector<std::vector<double>> {
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
  
  gem_roll_eta_limits_odd_l1 = new std::vector<std::pair<double,double> >{};
  gem_roll_eta_limits_odd_l2 = new std::vector<std::pair<double,double> >{};
  gem_roll_eta_limits_even_l1 = new std::vector<std::pair<double,double> >{};
  gem_roll_eta_limits_even_l2 = new std::vector<std::pair<double,double> >{};
  
  csc_wg_to_gem_roll_odd = new std::vector<std::pair<int,int> >{};
  
  csc_wg_to_gem_roll_even = new std::vector<std::pair<int,int> >{};
  
  gem_pad_to_csc_hs_me1a_odd = new std::vector<int>{
  };
  
  gem_pad_to_csc_hs_me1b_odd = new std::vector<int>{
  };
  
  gem_pad_to_csc_hs_me1a_even = new std::vector<int>{
  };
  
  gem_pad_to_csc_hs_me1b_even = new std::vector<int>{
  };
  
  // LUTs for positive endcap differ from negative endcap!
  csc_hs_to_gem_pad_me1a_odd = new std::vector<std::pair<int,int> >{
  };
  
  csc_hs_to_gem_pad_me1a_even = new std::vector<std::pair<int,int> >{
  };
  
  csc_hs_to_gem_pad_me1b_odd = new std::vector<std::pair<int,int> >{
  };
  
  csc_hs_to_gem_pad_me1b_even = new std::vector<std::pair<int,int> >{
  };
}

CSCGEMMotherboardLUTME11::~CSCGEMMotherboardLUTME11() {
 if (lut_wg_vs_hs_me1a) delete lut_wg_vs_hs_me1a;
 if (lut_wg_vs_hs_me1ag) delete lut_wg_vs_hs_me1ag;
 if (lut_wg_vs_hs_me1b) delete lut_wg_vs_hs_me1b;

 if (csc_wg_to_gem_roll_odd) delete csc_wg_to_gem_roll_odd;
 if (csc_wg_to_gem_roll_even) delete csc_wg_to_gem_roll_even;
 
 if (gem_pad_to_csc_hs_me1a_odd) delete gem_pad_to_csc_hs_me1a_odd;
 if (gem_pad_to_csc_hs_me1a_even) delete gem_pad_to_csc_hs_me1a_even;
 if (gem_pad_to_csc_hs_me1b_odd) delete gem_pad_to_csc_hs_me1b_odd;
 if (gem_pad_to_csc_hs_me1b_even) delete gem_pad_to_csc_hs_me1b_even;

 if (csc_hs_to_gem_pad_me1a_odd) delete csc_hs_to_gem_pad_me1a_odd;
 if (csc_hs_to_gem_pad_me1a_even) delete csc_hs_to_gem_pad_me1a_even;
 if (csc_hs_to_gem_pad_me1b_odd) delete csc_hs_to_gem_pad_me1b_odd;
 if (csc_hs_to_gem_pad_me1b_even) delete csc_hs_to_gem_pad_me1b_even;
}


CSCGEMMotherboardLUTME21::CSCGEMMotherboardLUTME21() :   CSCGEMMotherboardLUT()
{
  lut_wg_eta_odd = new std::vector<std::vector<double>> {
  };
  
  lut_wg_eta_even = new std::vector<std::vector<double>> {
  };
  
  lut_pt_vs_dphi_gemcsc = new std::vector<std::vector<double>> {
    {3, 0.01832829, 0.01003643 },
    {5, 0.01095490, 0.00631625 },
    {7, 0.00786026, 0.00501017 },
    {10, 0.00596349, 0.00414560 },
    {15, 0.00462411, 0.00365550 },
    {20, 0.00435298, 0.00361550 },
    {30, 0.00465160, 0.00335700 },
    {40, 0.00372145, 0.00366262 }
  };

  gem_roll_eta_limits_odd_l1 = new std::vector<std::pair<double,double> >{};
  gem_roll_eta_limits_odd_l2 = new std::vector<std::pair<double,double> >{};
  gem_roll_eta_limits_even_l1 = new std::vector<std::pair<double,double> >{};
  gem_roll_eta_limits_even_l2 = new std::vector<std::pair<double,double> >{};
  
  csc_wg_to_gem_roll_odd = new std::vector<std::pair<int,int> >{
  };

  csc_wg_to_gem_roll_even = new std::vector<std::pair<int,int> >{
  };
  
  gem_pad_to_csc_hs_odd = new std::vector<int>{
  };

  gem_pad_to_csc_hs_even = new std::vector<int>{
  };
  
  csc_hs_to_gem_pad_odd = new std::vector<std::pair<int,int> >{
  };

  csc_hs_to_gem_pad_even = new std::vector<std::pair<int,int> >{
  };
}


CSCGEMMotherboardLUTME21::~CSCGEMMotherboardLUTME21() 
{
 if (csc_wg_to_gem_roll_odd) delete csc_wg_to_gem_roll_odd;
 if (csc_wg_to_gem_roll_even) delete csc_wg_to_gem_roll_even;
 
 if (gem_pad_to_csc_hs_odd) delete gem_pad_to_csc_hs_odd;
 if (gem_pad_to_csc_hs_even) delete gem_pad_to_csc_hs_even;

 if (csc_hs_to_gem_pad_odd) delete csc_hs_to_gem_pad_odd;
 if (csc_hs_to_gem_pad_even) delete csc_hs_to_gem_pad_even;
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
  if(lut_wg_eta_odd) delete lut_wg_eta_odd;
  if(lut_wg_eta_even) delete lut_wg_eta_even;

  if(rpc_roll_eta_limits_odd) delete rpc_roll_eta_limits_odd;
  if(rpc_roll_eta_limits_even) delete rpc_roll_eta_limits_even;

  if(rpc_strip_to_csc_hs_odd) delete rpc_strip_to_csc_hs_odd;
  if(rpc_strip_to_csc_hs_even) delete rpc_strip_to_csc_hs_even;

  if(csc_hs_to_rpc_strip_odd) delete csc_hs_to_rpc_strip_odd;
  if(csc_hs_to_rpc_strip_even) delete csc_hs_to_rpc_strip_even;

  if(csc_wg_to_rpc_roll_odd) delete csc_wg_to_rpc_roll_odd;
  if(csc_wg_to_rpc_roll_even) delete csc_wg_to_rpc_roll_even;
}

CSCRPCMotherboardLUTME31::CSCRPCMotherboardLUTME31() 
  : CSCRPCMotherboardLUT()
{
  /*
const double CSCRPCMotherboard::lut_wg_me31_eta_odd[96][2] = {
{ 0,2.421},{ 1,2.415},{ 2,2.406},{ 3,2.397},{ 4,2.388},{ 5,2.379},{ 6,2.371},{ 7,2.362},
{ 8,2.353},{ 9,2.345},{10,2.336},{11,2.328},{12,2.319},{13,2.311},{14,2.303},{15,2.295},
{16,2.287},{17,2.279},{18,2.271},{19,2.263},{20,2.255},{21,2.248},{22,2.240},{23,2.232},
{24,2.225},{25,2.217},{26,2.210},{27,2.203},{28,2.195},{29,2.188},{30,2.181},{31,2.174},
{32,2.169},{33,2.157},{34,2.151},{35,2.142},{36,2.134},{37,2.126},{38,2.118},{39,2.110},
{40,2.102},{41,2.094},{42,2.087},{43,2.079},{44,2.071},{45,2.064},{46,2.056},{47,2.049},
{48,2.041},{49,2.034},{50,2.027},{51,2.019},{52,2.012},{53,2.005},{54,1.998},{55,1.991},
{56,1.984},{57,1.977},{58,1.970},{59,1.964},{60,1.957},{61,1.950},{62,1.944},{63,1.937},
{64,1.932},{65,1.922},{66,1.917},{67,1.911},{68,1.905},{69,1.898},{70,1.892},{71,1.886},
{72,1.880},{73,1.874},{74,1.868},{75,1.861},{76,1.855},{77,1.850},{78,1.844},{79,1.838},
{80,1.832},{81,1.826},{82,1.820},{83,1.815},{84,1.809},{85,1.803},{86,1.798},{87,1.792},
{88,1.787},{89,1.781},{90,1.776},{91,1.770},{92,1.765},{93,1.759},{94,1.754},{95,1.749},
};

const double CSCRPCMotherboard::lut_wg_me31_eta_even[96][2] = {
{ 0,2.447},{ 1,2.441},{ 2,2.432},{ 3,2.423},{ 4,2.414},{ 5,2.405},{ 6,2.396},{ 7,2.388},
{ 8,2.379},{ 9,2.371},{10,2.362},{11,2.354},{12,2.345},{13,2.337},{14,2.329},{15,2.321},
{16,2.313},{17,2.305},{18,2.297},{19,2.289},{20,2.281},{21,2.273},{22,2.266},{23,2.258},
{24,2.251},{25,2.243},{26,2.236},{27,2.228},{28,2.221},{29,2.214},{30,2.207},{31,2.200},
{32,2.195},{33,2.183},{34,2.176},{35,2.168},{36,2.160},{37,2.152},{38,2.144},{39,2.136},
{40,2.128},{41,2.120},{42,2.112},{43,2.104},{44,2.097},{45,2.089},{46,2.082},{47,2.074},
{48,2.067},{49,2.059},{50,2.052},{51,2.045},{52,2.038},{53,2.031},{54,2.023},{55,2.016},
{56,2.009},{57,2.003},{58,1.996},{59,1.989},{60,1.982},{61,1.975},{62,1.969},{63,1.962},
{64,1.957},{65,1.948},{66,1.943},{67,1.936},{68,1.930},{69,1.924},{70,1.917},{71,1.911},
{72,1.905},{73,1.899},{74,1.893},{75,1.887},{76,1.881},{77,1.875},{78,1.869},{79,1.863},
{80,1.857},{81,1.851},{82,1.845},{83,1.840},{84,1.834},{85,1.828},{86,1.823},{87,1.817},
{88,1.811},{89,1.806},{90,1.800},{91,1.795},{92,1.790},{93,1.784},{94,1.779},{95,1.774},
};
*/
}

CSCRPCMotherboardLUTME41::CSCRPCMotherboardLUTME41() 
  : CSCRPCMotherboardLUT()
{
  /*
const double CSCRPCMotherboard::lut_wg_me41_eta_odd[96][2] = {
{ 0,2.399},{ 1,2.394},{ 2,2.386},{ 3,2.378},{ 4,2.370},{ 5,2.362},{ 6,2.354},{ 7,2.346},
{ 8,2.339},{ 9,2.331},{10,2.323},{11,2.316},{12,2.308},{13,2.301},{14,2.293},{15,2.286},
{16,2.279},{17,2.272},{18,2.264},{19,2.257},{20,2.250},{21,2.243},{22,2.236},{23,2.229},
{24,2.223},{25,2.216},{26,2.209},{27,2.202},{28,2.196},{29,2.189},{30,2.183},{31,2.176},
{32,2.172},{33,2.161},{34,2.157},{35,2.150},{36,2.144},{37,2.138},{38,2.132},{39,2.126},
{40,2.119},{41,2.113},{42,2.107},{43,2.101},{44,2.095},{45,2.089},{46,2.083},{47,2.078},
{48,2.072},{49,2.066},{50,2.060},{51,2.055},{52,2.049},{53,2.043},{54,2.038},{55,2.032},
{56,2.027},{57,2.021},{58,2.016},{59,2.010},{60,2.005},{61,1.999},{62,1.994},{63,1.989},
{64,1.985},{65,1.977},{66,1.973},{67,1.968},{68,1.963},{69,1.958},{70,1.953},{71,1.947},
{72,1.942},{73,1.937},{74,1.932},{75,1.928},{76,1.923},{77,1.918},{78,1.913},{79,1.908},
{80,1.903},{81,1.898},{82,1.894},{83,1.889},{84,1.884},{85,1.879},{86,1.875},{87,1.870},
{88,1.866},{89,1.861},{90,1.856},{91,1.852},{92,1.847},{93,1.843},{94,1.838},{95,1.834},
};

const double CSCRPCMotherboard::lut_wg_me41_eta_even[96][2] = {
{ 0,2.423},{ 1,2.418},{ 2,2.410},{ 3,2.402},{ 4,2.394},{ 5,2.386},{ 6,2.378},{ 7,2.370},
{ 8,2.362},{ 9,2.355},{10,2.347},{11,2.339},{12,2.332},{13,2.324},{14,2.317},{15,2.310},
{16,2.302},{17,2.295},{18,2.288},{19,2.281},{20,2.274},{21,2.267},{22,2.260},{23,2.253},
{24,2.246},{25,2.239},{26,2.233},{27,2.226},{28,2.219},{29,2.213},{30,2.206},{31,2.199},
{32,2.195},{33,2.185},{34,2.180},{35,2.174},{36,2.168},{37,2.161},{38,2.155},{39,2.149},
{40,2.143},{41,2.137},{42,2.131},{43,2.125},{44,2.119},{45,2.113},{46,2.107},{47,2.101},
{48,2.095},{49,2.089},{50,2.084},{51,2.078},{52,2.072},{53,2.067},{54,2.061},{55,2.055},
{56,2.050},{57,2.044},{58,2.039},{59,2.033},{60,2.028},{61,2.023},{62,2.017},{63,2.012},
{64,2.008},{65,2.000},{66,1.996},{67,1.991},{68,1.986},{69,1.981},{70,1.976},{71,1.971},
{72,1.966},{73,1.961},{74,1.956},{75,1.951},{76,1.946},{77,1.941},{78,1.936},{79,1.931},
{80,1.926},{81,1.921},{82,1.917},{83,1.912},{84,1.907},{85,1.902},{86,1.898},{87,1.893},
{88,1.889},{89,1.884},{90,1.879},{91,1.875},{92,1.870},{93,1.866},{94,1.861},{95,1.857},
};
*/
}

std::vector<std::vector<double>> * CSCRPCMotherboardLUT::get_lut_wg_eta(bool isEven) const
{
  return isEven ? lut_wg_eta_even : lut_wg_eta_odd;
}

std::vector<std::pair<double,double> >* CSCRPCMotherboardLUT::get_rpc_roll_eta_limits(bool isEven) const
{
  return isEven ? rpc_roll_eta_limits_even : rpc_roll_eta_limits_odd;
}

std::vector<int>* CSCRPCMotherboardLUT::get_rpc_strip_to_csc_hs(bool isEven) const
{
  return isEven ? rpc_strip_to_csc_hs_even : rpc_strip_to_csc_hs_odd;
}

std::vector<std::pair<int,int> >* CSCRPCMotherboardLUT::get_csc_hs_to_rpc_strip(bool isEven) const
{
  return isEven ? csc_hs_to_rpc_strip_even : csc_hs_to_rpc_strip_odd;
}

std::vector<int>* CSCRPCMotherboardLUT::get_csc_wg_to_rpc_roll(bool isEven) const
{
  return isEven ? csc_wg_to_rpc_roll_even : csc_wg_to_rpc_roll_odd;
}
