//
// Test program that will read and write out 
// DAC settings, trim bitsa and maskbits for a
// readout link object.
//
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "PixelFECInterface/include/PixelFECInterface.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfigInterface.h"

using namespace pos;

int main(){
 
  //First lets do some tests of the PixelROCName class

  //std::string path(getenv("XDAQ_ROOT"));
  //path+="/pixel/PixelFEDInterface/test/";
  //PixelFEDCard card(path+"params_fed.dat");

  //PixelCalibConfiguration calib("scan_calib.dat");

  //PixelDetectorConfig detconfig("detconfig.dat");

  PixelNameTranslation nametranslation("translation.dat");

  
  
  
  for(unsigned int fednumber=1; fednumber<2; fednumber++){
    for (unsigned int channel=1; channel<7; ++channel) {
      std::cout << "Will check fednumber="<<fednumber<<" and channel="<<channel<<std::endl;
      std::vector<PixelROCName> rocs=nametranslation.getROCsFromFEDChannel(fednumber, channel);
      for (unsigned int iroc=0; iroc<rocs.size(); ++iroc) {
        std::cout<<"ROC Name under FED ID "<<fednumber<<" channel="<<channel<<" ROC Name="<<rocs[iroc]<<std::endl;
      }
    }
  }

  


  return 0;

}
