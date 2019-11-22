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
#include "CalibFormats/SiPixelObjects/interface/PixelPortcardMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTKFECConfig.h"
#include "PixelUtilities/PixelTestStandUtilities/include/PixelTimer.h"

using namespace pos;
using namespace std;

int main() {
  //First lets do some tests of the PixelROCName class

  //std::string path(std::getenv("XDAQ_ROOT"));
  //path+="/pixel/PixelFEDInterface/test/";
  //PixelFEDCard card(path+"params_fed.dat");

  //PixelCalibConfiguration calib("calib.dat_orig");

  //calib.writeASCII("");

  //return 0;

  cout << "Will open translation.dat" << endl;

  PixelTimer t, t2, t3, t4;

  t2.start();

  PixelNameTranslation nametranslation("translation.dat");

  t2.stop();
  cout << "Time to read name translations:" << t2.tottime() << endl;

  t3.start();

  PixelCalibConfiguration calib("calib.dat");

  t3.stop();
  cout << "Time to read calib.dat:" << t3.tottime() << endl;

  t4.start();

  PixelDetectorConfig detconfig("detectconfig.dat");

  t4.stop();
  cout << "Time to read detectconfig:" << t4.tottime() << endl;

  PixelPortcardMap portcardmap("portcardmap.dat");

  PixelFECConfig fecconfig("fecconfig.dat");

  PixelTKFECConfig tkfecconfig("tkfecconfig.dat");

  const std::set<std::string>& portcards = portcardmap.portcards(&detconfig);

  cout << "Will start to build ROC list" << endl;

  t.start();

  calib.buildROCAndModuleLists(&nametranslation, &detconfig);

  t.stop();

  cout << "Done with building ROC list" << endl;

  cout << "Size of ROC list:" << calib.rocList().size() << endl;

  cout << "Time to build ROC list:" << t.tottime() << endl;

  /*
  
  
  for(unsigned int fednumber=1; fednumber<2; fednumber++){
    for (unsigned int channel=1; channel<7; ++channel) {
      std::cout << "Will check fednumber="<<fednumber<<" and channel="<<channel<<std::endl;
      std::vector<PixelROCName> rocs=nametranslation.getROCsFromFEDChannel(fednumber, channel);
      for (unsigned int iroc=0; iroc<rocs.size(); ++iroc) {
        std::cout<<"ROC Name under FED ID "<<fednumber<<" channel="<<channel<<" ROC Name="<<rocs[iroc]<<std::endl;
      }
    }
  }
  */

  return 0;
}
