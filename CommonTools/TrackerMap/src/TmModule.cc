#include "CommonTools/TrackerMap/interface/TmModule.h"
#include <string>
using namespace std;


map< const int  , TmModule *>
SvgModuleMap::smoduleMap=map<const int  , TmModule *>();

map< const int  , TmModule *>
IdModuleMap::imoduleMap=map<const int  , TmModule *>();


TmModule::TmModule(int idc, int ring, int layer){
  idModule = idc;
  this->ring=ring;
  this->layer = layer;
  this->text="";
 notused=true;
 histNumber=0;
 red=-1;
}
TmModule::~TmModule(){
}
