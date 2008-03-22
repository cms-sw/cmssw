#include "CommonTools/TrackerMap/interface/TmApvPair.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include <string>
using namespace std;

TmApvPair::TmApvPair(int connId){
  idex=connId;
  value=0;count=0;
  red = -1;
}

TmApvPair::~TmApvPair(){
}

