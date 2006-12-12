#include "CommonTools/TrackerMap/interface/TmApvPair.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include <string>
using namespace std;

map< const int  , TmApvPair *>
SvgApvPair::apvMap=map<const int  , TmApvPair *>();
map< const int  , int>
SvgFed::fedMap=map<const int  , int>();

TmApvPair::TmApvPair(int connId){
  idex=connId;
  value=0;count=0;
  red = -1;
  histNumber=0;
}

TmApvPair::~TmApvPair(){
}

