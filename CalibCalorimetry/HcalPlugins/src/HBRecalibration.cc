///////////////////////////////////////////////////////////////////////////////
// File: HBRecalibration.cc
// Description: simple helper class containing tabulater/parameterized 
//              function for HE damage recovery for Upgrade studies  
//              evaluated using DataFormats/HcalCalibObjects/HBDarkening   
///////////////////////////////////////////////////////////////////////////////

#include "HBRecalibration.h"

HBRecalibration::HBRecalibration(double integrated_lumi, double cutoff, unsigned int scenario):
iLumi(integrated_lumi),cutoff_(cutoff),darkening(scenario)
{ }
   
HBRecalibration::~HBRecalibration() { }

void HBRecalibration::setDsegm( const std::vector<std::vector<int>>& m_segmentation) 
{

  //  std::cout << std::endl << " HBRecalibration->setDsegm" << std::endl;

  for (unsigned int ieta = 0; ieta < HBDarkening::nEtaBins; ieta++) {
    //    std::cout << "["<< ieta << "]  ieta =" << ieta + 16 << "  ";
    for(unsigned int ilay = 0; ilay < HBDarkening::nScintLayers; ilay++) {
      dsegm[ieta][ilay] = m_segmentation[ieta+1][ilay]; // 0 not used
      //      std::cout << dsegm [ieta][ilay];
    }
    //    std::cout << std::endl;
  }

  initialize();

}

double HBRecalibration::getCorr(int ieta, int idepth) {

  //  int init_ieta = ieta;
  ieta = abs(ieta)-1; // 0 - 15
  
  if(corr[ieta][idepth] > cutoff_) return cutoff_;
  else return corr[ieta][idepth];
}


void HBRecalibration::initialize() {

  double dval[HBDarkening::nEtaBins][nDepths];  // conversion of lval into depths-averaged values - denominator (including degradation for iLumi) 
  double nval[HBDarkening::nEtaBins][nDepths];  // conversion of lval into depths-averaged values - numerator (no degradation)

  for (unsigned int j = 0; j < HBDarkening::nEtaBins; j++) {
    for (unsigned int k = 0; k < nDepths; k++) {
      dval[j][k] = 0.0;
      nval[j][k] = 0.0;
    }
  }

  double lval[HBDarkening::nEtaBins][HBDarkening::nScintLayers]    // raw table of mean energy in each layer for each ieta at 0 lumi
    = {
      {2.362808,1.575159,1.283007,1.026073,0.834189,0.702393,0.566008,0.484473,0.402106,0.306254,0.251159,0.199382,0.156932,0.132067,0.099506,0.080853,0.118480}, //tower 1
      {2.397443,1.616431,1.301160,1.078375,0.882232,0.705615,0.577277,0.472366,0.383500,0.326740,0.265406,0.208601,0.169150,0.124831,0.103368,0.078420,0.117037}, //tower 2
      {2.475831,1.654173,1.322021,1.033156,0.850223,0.684293,0.551837,0.467637,0.377493,0.329712,0.254339,0.203073,0.165079,0.123208,0.102514,0.079879,0.112186}, //tower 3
      {2.462020,1.606079,1.282836,1.010049,0.855179,0.669815,0.556634,0.452061,0.377653,0.311965,0.255425,0.194654,0.159325,0.117479,0.092718,0.069317,0.105891}, //tower 4
      {2.707523,1.747932,1.373208,1.106487,0.866754,0.728357,0.563662,0.469271,0.374237,0.303459,0.241259,0.187083,0.140809,0.116930,0.081804,0.064648,0.101095}, //tower 5
      {2.678347,1.741005,1.317389,1.013474,0.823785,0.710703,0.551169,0.429373,0.347780,0.274428,0.222277,0.166950,0.131702,0.101153,0.075034,0.060390,0.092811}, //tower 6
      {2.809996,1.729876,1.328158,1.050613,0.820268,0.669393,0.516984,0.411677,0.331940,0.272142,0.197038,0.155147,0.127178,0.094173,0.069392,0.054302,0.071747}, //tower 7
      {2.858155,1.770711,1.355659,1.047950,0.851036,0.657216,0.526521,0.416379,0.319870,0.247920,0.188156,0.145095,0.110619,0.080743,0.063796,0.052655,0.071961}, //tower 8
      {3.041316,1.877900,1.418290,1.082000,0.840637,0.676057,0.505557,0.402506,0.301596,0.231630,0.182558,0.138136,0.106255,0.073500,0.056109,0.041644,0.045892}, //tower 9
      {3.142461,1.817359,1.363827,1.013841,0.768494,0.603310,0.463155,0.368469,0.282965,0.218877,0.152383,0.118167,0.079790,0.053056,0.038893,0.031578,0.040494}, //tower 10
      {3.294945,1.854570,1.367346,1.008908,0.769117,0.594254,0.445583,0.335300,0.248729,0.190494,0.137710,0.099664,0.071549,0.053245,0.037729,0.025820,0.033520}, //tower 11
      {3.579348,1.951865,1.393643,1.009883,0.745045,0.555898,0.424502,0.313736,0.220994,0.156913,0.109682,0.074274,0.052242,0.039518,0.029820,0.016796,0.028697}, //tower 12
      {3.752190,2.001947,1.431959,1.057370,0.750016,0.550919,0.410243,0.300922,0.201059,0.151015,0.111034,0.074270,0.049154,0.034618,0.025749,0.018542,0.025501}, //tower 13
      {4.057676,2.074200,1.423406,1.000701,0.745990,0.536405,0.375057,0.278802,0.192733,0.128711,0.094431,0.062570,0.051008,0.034336,0.026740,0.015083,0.002232}, //tower 14
      {4.252095,2.160939,1.492043,1.021666,0.740503,0.534830,0.378022,0.276588,0.204727,0.156732,0.111826,0.066944,0.045150,0.031974,0.019658,0.001816,0.000000}, //tower 15
      {0.311054,0.185613,2.806550,1.810331,0.996787,0.539745,0.285648,0.136337,0.059499,0.007867,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000} //tower 16
    };
  
  // coverting energy values from layers into depths 

  //  std::cout << std::endl << " >>> DVAL evaluation " << std::endl;

  for (unsigned int ieta = 0; ieta < HBDarkening::nEtaBins; ieta++) {
  
    //fill sum(means(layer,0)) and sum(means(layer,lumi)) for each depth
    for(unsigned int ilay = 0; ilay < HBDarkening::nScintLayers; ilay++) {
      int idepth = dsegm[ieta][ilay]; // idepth = 0 - not used!
      nval[ieta][idepth] += lval[ieta][ilay];
      dval[ieta][idepth] += lval[ieta][ilay]*darkening.degradation(iLumi,ieta+1,ilay); //be careful of eta and layer numbering
      
      /*
      std::cout << "ilay " << ilay << " -> idepth " << idepth   
              << "  + lval[" << ieta << "][" << ilay << "]"
       << " " << lval[ieta][ilay] <<  std::endl;       
      */
    }
   
    //compute factors, w/ safety checks
    for(unsigned int idepth = 0; idepth < nDepths; idepth++){
      if(dval[ieta][idepth] > 0) corr[ieta][idepth] = nval[ieta][idepth]/dval[ieta][idepth];
      else corr[ieta][idepth] = 1.0;
      
      if(corr[ieta][idepth] < 1.0) corr[ieta][idepth] = 1.0;
      /*
	if (idepth > 0 && idepth <= 3) {
	std::cout << "nval[" << ieta << "][" << idepth << "]"
	<< " = " << nval[ieta][idepth] << " - "
	  	    << "dval["<< ieta << "][" << idepth << "]"
	  	    << " = " << dval[ieta][idepth] 
		    << "    corr = " << corr[ieta][idepth] << std::endl;
		    }
      */
    }
    
  }


}
