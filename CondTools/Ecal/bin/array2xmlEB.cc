#include <iostream>
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <string>
#include <sstream>
#include <fstream>

using namespace std;

void usage(){
  cout << endl;
  cout << "array2xml [arrayfile] [xmlfile]" <<  endl;
  cout << "Read coefficients from straight array [denseindex]"<<endl;
  cout << "and write in xml format" << endl;

}


int main(int argc, char* argv[]){

 if (argc!=3) {
    usage();
    exit(0);
  }


 string arrayfilename(argv[1]);
 string xmlfilename(argv[2]);
 fstream arrayfile(arrayfilename.c_str(),ios::in);
 

 EcalIntercalibConstants rcd ;

 float c=0;
 int   idx=0;

 while (arrayfile>>c){
   uint32_t id=EBDetId::unhashIndex(idx);
   rcd[id]=c;
   ++idx;
 }
 cout << idx << endl;
 
 for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EB cells
    
    

    if (EEDetId::validHashIndex(cellid)){  
      uint32_t rawid = EEDetId::unhashIndex(cellid);
     
      rcd[rawid]=0.0;
     
    } // if
  } 



 
   
 // write new format
 EcalCondHeader h; 
 EcalIntercalibConstantsXMLTranslator::writeXML(xmlfilename,h,rcd); 
   

}
