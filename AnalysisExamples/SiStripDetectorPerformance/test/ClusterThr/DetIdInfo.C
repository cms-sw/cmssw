/**

$Date: 2007/01/19 11:38:15 $
$Revision: 1.5 $

author: D. Giordano, domenico.giordano@cern.ch
**/

#include "iostream"
#include "fstream"
#include "sstream"
#include "vector"
#include "map"
#include "string"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

class DetIdInfo{

public:
  DetIdInfo(){}
  ~DetIdInfo(){};
  
  void setDetId(unsigned int& d){detid=d;}
  void getInfo();

  int layer;
private:
  unsigned int detid;

  

};

void DetIdInfo::getInfo(){
  SiStripDetId a(detid);
  if (a.det()!=DetId::Tracker)
    return;

  std::cout << a << std::endl;

  if ( a.subdetId() == 3 ){
    TIBDetId b(detid);
    std::cout << "TIB (layer, string, ) " <<  b.layer() << " " << b.string()[0] << " " << b.string()[1] << " " << b.string()[2] << " " << b.glued() << " " << b.module()<< std::endl;
  } else if ( a.subdetId() == 4 ) {
    TIDDetId b(detid);
    std::cout << "TID (wheel, ring, side, glued, stereo)  " <<  b.wheel() << " " << b.ring() << " " << b.side() << " " << b.glued() << " " << b.stereo() << std::endl;
  } else if ( a.subdetId() == 5 ) {
    TOBDetId b(detid);
    std::cout << "TOB (layer, rod, ) " <<  b.layer() << " " << b.rod()[0] << " " << b.rod()[1] << " " << b.glued() << " " << b.module()<< std::endl;
  } else if ( a.subdetId() == 6 ) {
    TECDetId b(detid);
    std::cout << "TEC (wheel, ring, side, ) " <<  b.wheel() << " " << b.ring() << " " << b.side() << " " << b.glued() << " " << b.stereo() << std::endl;
    }  

}

