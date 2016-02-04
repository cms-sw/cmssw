//-------------------------------------------------
//
//   Class: L1MuGMTEtaLUT
//
//   Description: Look-up table for GMT Eta Projection Unit
//
//
//   $Date: 2007/03/23 18:51:35 $
//   $Revision: 1.3 $
//
//   Author :
//   H. Sakulin                CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTEtaLUT.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <vector>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// --------------------------------
//       class L1MuGMTEtaLUT
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuGMTEtaLUT::L1MuGMTEtaLUT() {
}

//--------------
// Destructor --
//--------------
L1MuGMTEtaLUT::~L1MuGMTEtaLUT() { 
  
}

//--------------
// Operations --
//--------------
 
int L1MuGMTEtaLUT::etabin(float eta, int isys) {
  int i;
  for (i=0; i<(int)NETA;i++)
    if (eta >= etabins[isys][i] && eta < etabins[isys][i+1]) break;
  if (i>=(int)NETA) edm::LogWarning("LUTProblem") << "L1MuGMTEtaLUT::etabin(): could not assign eta bin ";
  return i;
}


//
// project muon eta to calo/vertex
//
float L1MuGMTEtaLUT::eta(int isys, int isISO, int icharge, float eta, float pt) {

  int ieta = etabin ( (float) fabs(eta), isys );
  
  if ( (isys==DT || isys==bRPC) && pt < 4.) pt = 4.; // cut off
  if ( (isys==CSC || isys==fRPC) && pt < 3.) pt = 3.;

  float a=fitparams_eta[isISO][isys][ieta][0];
  float b=fitparams_eta[isISO][isys][ieta][1];
  float c=fitparams_eta[isISO][isys][ieta][2];

  float deta = a + b / pt + c / (pt*pt);
  
  float neweta;
  if (eta>0) neweta = eta - deta;
  else neweta = eta + deta;

  return neweta;
 }


//
// static parameters of LUTs
//

// 06/2003 changed a few end-of scale values 
// in order to get less errors when generating LUTs

float L1MuGMTEtaLUT::etabins[L1MuGMTEtaLUT::NSYS][L1MuGMTEtaLUT::NETA+1] = {
  { 0.00, 0.22, 0.27, 0.58, 0.77, 0.87, 0.92, 1.24, 1.35 /*1.24*/ }, // DT
  { 0.00, 1.06, 1.26, 1.46, 1.66, 1.86, 2.06, 2.26, 2.46 }, // CSC
  { 0.00, 0.06, 0.25, 0.41, 0.54, 0.70, 0.83, 0.93, 2.10 /* 1.04 */ }, // bRPC
  { 0.00 /*1.04*/ , 1.24, 1.36, 1.48, 1.61, 1.73, 1.85, 1.97, 2.10}   // fRPC
};


float L1MuGMTEtaLUT::fitparams_eta[L1MuGMTEtaLUT::NRP][L1MuGMTEtaLUT::NSYS][L1MuGMTEtaLUT::NETA][3]= {
  {
    // projection to HCAL derived from HCAL positions retrieved in ORCA
    // calo eta projection parametrization for DT, projection to HCAL in (for MIP)
    {
      {  0.000622, -0.041158,  0.173116  },
      {  0.006699,  0.033410,  0.100972  },
      { -0.013232,  0.067651, -0.601139  },
      { -0.014180,  0.021067,  0.523313  },
      {  0.016339, -0.088452,  0.760254  },
      { -0.036318, -0.191343,  2.020133  },
      {  0.034405, -0.085665,  1.199408  },
      {  0.000000,  0.000000,  0.000000  }
    },
    // calo eta projection parametrization for CSC, projection to HCAL in (for MIP)
    {
      { -0.015401, -0.058141,  1.558118  },
      { -0.004330, -0.070188,  1.284233  },
      { -0.006400, -0.087547,  1.504680  },
      { -0.012328, -0.017183,  0.740447  },
      { -0.008242,  0.065439,  0.103247  },
      { -0.007771,  0.064422,  0.032815  },
      { -0.013519,  0.061566, -0.012504  },
      { -0.019427,  0.058284, -0.137946  }
    },
    // calo eta projection parametrization for bRPC, projection to HCAL in (for MIP)
    {
      { -0.007561, -0.042023,  0.352816  },
      { -0.009209,  0.040731,  0.302872  },
      { -0.009222,  0.187243, -0.778960  },
      {  0.005989,  0.166652, -0.389902  },
      {  0.002763,  0.015457,  0.659074  },
      {  0.003961,  0.024581,  0.356591  },
      {  0.007231, -0.006835,  0.780842  },
      {  0.009003, -0.096096,  1.400512  }
    },
    // calo eta projection parametrization for fRPC, projection to HCAL in (for MIP)
    {
      {  0.001764, -0.005724,  0.996918  },
      { -0.001087, -0.006683,  1.025424  },
      {  0.006963, -0.070388,  1.071490  },
      { -0.010473, -0.009773,  0.757894  },
      { -0.003982,  0.044828,  0.188175  },
      { -0.004707,  0.090827,  0.075001  },
      { -0.008527,  0.093006, -0.030012  },
      { -0.002392,  0.070246,  0.021308  }
    }
  },
  {
    // calo eta projection parametrization for DT, projection to vertex (for ISO)
    {
      { -0.000091, -0.017684,  0.055423  },
      { -0.001020, -0.043767,  0.895337  },
      { -0.012845,  0.042033, -0.260102  },
      { -0.013225,  0.023254,  0.711623  },
      {  0.019014, -0.106147,  1.048216  },
      { -0.037251, -0.186894,  2.284707  },
      {  0.032260, -0.064396,  1.457979  },
      {  0.000000,  0.000000,  0.000000  }
    },
    // calo eta projection parametrization for CSC, projection to vertex (for ISO)
    {
      { -0.005587, -0.055360,  1.573220  },
      { -0.005393, -0.048236,  1.591642  },
      { -0.006649, -0.091712,  1.716567  },
      { -0.007636, -0.061966,  1.065366  },
      { -0.000338, -0.020505,  0.578064  },
      { -0.001077, -0.015956,  0.384573  },
      { -0.001851, -0.029931,  0.310996  },
      { -0.001878, -0.012289,  0.168514  }
    },
    // calo eta projection parametrization for bRPC, projection to vertex (for ISO)
    {
      { -0.010030,  0.027686, -0.147926  },
      { -0.008563,  0.041026,  0.356772  },
      { -0.010788,  0.166841, -0.458204  },
      {  0.009806,  0.109100,  0.080603  },
      {  0.000945,  0.028016,  0.798682  },
      {  0.006432,  0.007630,  0.704437  },
      {  0.007685, -0.004569,  1.044223  },
      {  0.005465, -0.063151,  1.631665  }
    },
    // calo eta projection parametrization for fRPC, projection to vertex (for ISO)
    {
      {  0.003319, -0.027350,  1.453144  },
      {  0.002237, -0.038884,  1.527818  },
      {  0.003157, -0.092544,  1.479619  },
      { -0.002126, -0.074455,  1.120056  },
      { -0.002724, -0.011819,  0.628300  },
      {  0.002581, -0.017028,  0.607589  },
      {  0.002323,  0.025784,  0.253405  },
      {  0.006582,  0.008247,  0.258189  }
    }
  }
};

