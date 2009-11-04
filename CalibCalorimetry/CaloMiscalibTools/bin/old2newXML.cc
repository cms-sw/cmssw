#include <iostream>
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"

#include <string>
#include <sstream>

using namespace std;

void usage(){
  cout << endl;
  cout << "old2newXML [oldfileeb]  [oldfileee] [outfile] " <<  endl;
  cout << "Read coefficients from  files in old xml format"<<endl;
  cout << " <Cell ieta iphi scale_factor /> " << endl;
  cout << " and translate to new format " << endl;  
}


int main(int argc, char* argv[]){

 if (argc!=4) {
    usage();
    exit(0);
  }


 string barrelfile(argv[1]);
 string endcapfile(argv[2]);
 string outfile(argv[3]);

 // read old format

 CaloMiscalibMapEcal cmap;
 
 cmap.prefillMap();
 MiscalibReaderFromXMLEcalBarrel barrelreader(cmap);
 MiscalibReaderFromXMLEcalEndcap endcapreader(cmap);
 barrelreader.parseXMLMiscalibFile(barrelfile);
 endcapreader.parseXMLMiscalibFile(endcapfile);

 const EcalIntercalibConstants& rcd = cmap.get() ;
   
 // write new format
 EcalCondHeader h; 
 EcalIntercalibConstantsXMLTranslator::writeXML(outfile,h,rcd); 
   

}
