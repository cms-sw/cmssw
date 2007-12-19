#include "CommonTools/TrackerMap/interface/TmApvPair.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include <string>
using namespace std;

TmApvPair::TmApvPair(int connId,int crate){
  idex=connId;
  this->crate=crate;
  value=0;count=0;
  red = -1;
  text="";
}

TmApvPair::~TmApvPair(){
}

