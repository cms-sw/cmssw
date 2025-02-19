#include "CommonTools/TrackerMap/interface/TmPsu.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include <string>

TmPsu::TmPsu(int dcs,int branch,int rack, int crate,int board){
  
  id=dcs*100000+branch*1000+crate*100+board;
  idex=rack*1000+crate*100+board;
  value=0;count=0;
  countHV2=0;
  countHV3=0;
  valueHV2=0;
  valueHV3=0;
  red = -1;
  redHV2 = -1;
  redHV3 = -1;
  text="";
  textHV2="";
  textHV3="";


}

TmPsu::~TmPsu(){
}
