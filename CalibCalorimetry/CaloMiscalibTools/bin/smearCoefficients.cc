#include <iostream>
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <string>
#include <sstream>
#include "TRandom.h"

using namespace std;

void usage(){
  cout << endl;
  cout << "smearCoefficients [infile] [outfile] [barrelsmear] [endcapsmear]" 
       <<  endl;
  cout << "Read coefficients from  [infile] in xml format,"
       << " smear with a gaussian of mean 1 and sigma" 
       << "[barrelsmear]/[endcapsmear] and write to [outfile]" << endl;
  cout << "Use EcalCondDB.py to dump constant sets" << endl;
}


int main(int argc, char* argv[]){

  if (argc!=5) {
    usage();
    exit(0);
  }

  // input parameters
  string infile(argv[1]);
  string outfile(argv[2]);
  string smeareb_s(argv[3]);
  string smearee_s(argv[4]);


  double smeareb=0;
  double smearee=0;
  
  stringstream tmp,tmp2; 
  tmp <<smeareb_s; 
  tmp>> smeareb;   
  tmp2 << smearee_s; 
  tmp2>> smearee;  

  if (smeareb==0 || smearee==0) {
    cout << "Smearing cannot be zero" << endl;
    exit(1);
  }

  // read 
  EcalIntercalibConstants rcd;
  EcalCondHeader h;
  
  int ret = EcalIntercalibConstantsXMLTranslator::readXML(infile,h,rcd);
  if (ret) {
    cout<< "Problems with file " << infile<< endl;
    exit(1);
  }
  
  TRandom rnd;
  // smear barrel
  for (int cellid = 0; 
	     cellid < EBDetId::kSizeForDenseIndexing; 
	     ++cellid){// loop on EB cells
    
    uint32_t rawid = EBDetId::unhashIndex(cellid);
    float smearing = rnd.Gaus(1,smeareb);
    rcd[rawid]= rcd[rawid]*smearing;
  } 

    

  // smear endcap
   
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EE cells
    
    

    if (EEDetId::validHashIndex(cellid)){  
      uint32_t rawid = EEDetId::unhashIndex(cellid);
      float smearing = rnd.Gaus(1,smearee);
      rcd[rawid]= rcd[rawid]*smearing;

    } // if
  } 


  EcalIntercalibConstantsXMLTranslator::writeXML(outfile,h,rcd);
  
}
