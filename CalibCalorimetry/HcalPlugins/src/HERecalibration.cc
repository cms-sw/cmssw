///////////////////////////////////////////////////////////////////////////////
// File: HERecalibration.cc
// Description: simple helper class containing tabulater/parameterized 
//              function for HE damade recovery for Upgrade studies  
//              evaluated using SimG4CMS/Calo/ HEDarkening   
///////////////////////////////////////////////////////////////////////////////

#include "HERecalibration.h"

HERecalibration::HERecalibration(double integrated_lumi):iLumi(integrated_lumi), darkening()
{ }
   
HERecalibration::~HERecalibration() { }

void HERecalibration::setDsegm( std::vector<std::vector<int>> m_segmentation) 
{

  //  std::cout << std::endl << " HERecalibration->setDsegm" << std::endl;

  for (int ieta = 0; ieta < maxEta; ieta++) {
    //    std::cout << "["<< ieta << "]  ieta =" << ieta + 16 << "  ";
    for(int ilay = 0; ilay < maxLay; ilay++) {
      dsegm[ieta][ilay] = m_segmentation[ieta+15][ilay]; // 0 not used
      //      std::cout << dsegm [ieta][ilay];
    }
    //    std::cout << std::endl;
  }

  initialize();

}

double HERecalibration::getCorr(int ieta, int idepth) {

  //  int init_ieta = ieta;
  ieta = abs(ieta)-16; // 0 - 13   
  double cutoff = 20.0;  // cutoff to avoid too big corrections!
  
  if(corr[ieta][idepth] > cutoff) return cutoff;
  else return corr[ieta][idepth];
}


void HERecalibration::initialize() {

  double dval[maxEta][maxDepth];  // conversion of lval into depths-averaged values - denominator (including degradation for iLumi) 
  double nval[maxEta][maxDepth];  // conversion of lval into depths-averaged values - numerator (no degradation)

  for (int j = 0; j < maxEta; j++) {
    for (int k = 0; k < maxDepth; k++) {
      dval[j][k] = 0.0;
      nval[j][k] = 0.0;
    }
  }

  double lval[maxEta][maxLay]    // raw table of mean energy in each layer for each ieta at 0 lumi
    = {
      {0.000000,0.000000,0.001078,0.008848,0.014552,0.011611,0.008579,0.003211,0.002964,0.001775,0.001244,0.000194,0.000159,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000}, //tower 16
      {0.107991,0.110127,0.081192,0.050552,0.032968,0.022363,0.012158,0.009392,0.006228,0.003650,0.003512,0.001384,0.002693,0.000171,0.000012,0.000000,0.000000,0.000000,0.000000}, //tower 17
      {0.676505,0.490168,0.296676,0.171789,0.129949,0.082862,0.058659,0.044634,0.031707,0.019662,0.015764,0.009106,0.006647,0.004244,0.003099,0.002173,0.001148,0.001037,0.001255}, //tower 18
      {0.000000,1.722120,1.182322,0.777626,0.551815,0.381650,0.287366,0.209747,0.149539,0.095313,0.072690,0.052678,0.039654,0.026296,0.017126,0.010785,0.009125,0.006478,0.005883}, //tower 19
      {0.000000,2.253261,1.735958,1.188160,0.840946,0.598602,0.468244,0.302274,0.231664,0.155488,0.107260,0.071025,0.052279,0.040206,0.029258,0.016750,0.013793,0.010577,0.005453}, //tower 20
      {0.000000,2.531237,1.890250,1.299543,0.910114,0.669070,0.488073,0.362763,0.261798,0.177501,0.126352,0.089263,0.060376,0.045327,0.028015,0.021607,0.014022,0.010632,0.007277}, //tower 21
      {0.000000,2.864449,2.128851,1.430183,0.997971,0.742801,0.534812,0.386734,0.272178,0.202083,0.151007,0.106616,0.080018,0.055961,0.042164,0.026671,0.017066,0.010679,0.008012}, //tower 22
      {0.000000,3.245071,2.455721,1.654948,1.168912,0.847157,0.620316,0.450344,0.329651,0.237315,0.164873,0.111421,0.076739,0.058448,0.042908,0.026116,0.019845,0.012941,0.009722}, //tower 23
      {0.000000,3.523457,2.661981,1.771658,1.309808,0.945457,0.701592,0.484851,0.369293,0.265011,0.187915,0.131769,0.095761,0.066367,0.047637,0.034704,0.025890,0.019738,0.011549}, //tower 24
      {0.000000,3.927840,3.003811,2.029410,1.449478,1.099043,0.807025,0.585442,0.438498,0.318257,0.227604,0.153886,0.115857,0.088632,0.057335,0.040283,0.031698,0.022189,0.013614}, //tower 25
      {0.000000,4.351642,3.468444,2.368010,1.716175,1.279396,0.944951,0.700572,0.508232,0.371673,0.274277,0.195301,0.136719,0.100344,0.081408,0.057241,0.039744,0.027569,0.017390}, //tower 26
      {0.000000,3.315232,2.799176,1.928155,1.384991,1.040579,0.757341,0.574996,0.419020,0.306780,0.219979,0.160667,0.112313,0.084583,0.068486,0.052088,0.034637,0.023234,0.018230}, //tower 27
      {0.000000,3.960416,3.689421,2.514970,1.835407,1.372355,1.447654,1.087639,0.836911,0.652146,0.507556,0.416950,0.333755,0.283386,0.224336,0.191078,0.169798,0.144836,0.120804}, //tower 28
      {0.000000,1.530837,1.507162,0.977401,0.694320,0.543973,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000} //tower 29
    };
  
  // coverting energy values from layers into depths 

  //  std::cout << std::endl << " >>> DVAL evaluation " << std::endl;

  for (int ieta = 0; ieta < maxEta; ieta++) {
  
    //fill sum(means(layer,0)) and sum(means(layer,lumi)) for each depth
    for(int ilay = 0; ilay < maxLay; ilay++) {
      int idepth = dsegm[ieta][ilay]; // idepth = 0 - not used!
      nval[ieta][idepth] += lval[ieta][ilay];
      dval[ieta][idepth] += lval[ieta][ilay]*darkening.degradation(iLumi,ieta+16,ilay-1); //be careful of eta and layer numbering
      
      /*
      std::cout << "ilay " << ilay << " -> idepth " << idepth   
              << "  + lval[" << ieta << "][" << ilay << "]"
       << " " << lval[ieta][ilay] <<  std::endl;       
      */
    }
   
    //compute factors, w/ safety checks
	for(int idepth = 0; idepth < maxDepth; idepth++){
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
