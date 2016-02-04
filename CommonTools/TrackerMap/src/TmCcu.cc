#include "CommonTools/TrackerMap/interface/TmCcu.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include <string>

TmCcu::TmCcu(int crate,int slot,int ring, int addr){
  idex=crate*10000000+slot*100000+ring*1000+addr;
  this->crate=crate;
  value=0;count=0;
  red = -1;
  text="";
  nmod=0;
  cmodid="";
  layer=0;
}

TmCcu::~TmCcu(){
}

