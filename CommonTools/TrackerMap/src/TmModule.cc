#include "CommonTools/TrackerMap/interface/TmModule.h"
#include <string>

TmModule::TmModule(int idc, int ring, int layer){
  idModule = idc;
  this->ring=ring;
  this->layer = layer;
  this->text="";
 notused=true;
 histNumber=0;
 red=-1;
 capvids="";
 CcuId=0;
 HVchannel=2;
}
TmModule::~TmModule(){
}
