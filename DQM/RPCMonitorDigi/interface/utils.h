#ifndef rpcdqm_utils_H
#define rpcdqm_utils_H

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

namespace rpcdqm{
  class utils{
  public:
    int detId2RollNr(const RPCDetId & _id){
      if(_id.region()==0){//Barrel
	if( _id.station() ==1) {// Station1
	  if(_id.layer() ==1) { //in
	    if(_id.roll()==1) //forward
	      _cnr = 1;//RB1inF
	    else //backward
	      _cnr = 2;//RB1inB
	  } else {//out
	    if(_id.roll()==1) 
	      _cnr = 3;//RB1outF
	    else 
	      _cnr = 4;//RB1outB 
	  }
	}else if (_id.station()==2) {//Station 2	
	  if(_id.layer()==1) {
	    if(_id.roll()==1)
	      _cnr = 5;//RB2inF
	    else if(_id.roll()==3)
	      _cnr=6;//RB2inM
	    else if(_id.roll()==2)
	      _cnr=7;//RB2inB
	  }else{
	    if(_id.roll()==2)
	      _cnr=7;
	    else if(_id.roll()==1)
	      _cnr=8;
	    else if(_id.roll()==3)
	      _cnr=9;	  
	  }
	}else  if(_id.station()==3){//Station 3
	  if(_id.subsector()==1){
	    if(_id.roll()==1)
	      _cnr=10;
	    else 
	      _cnr=11;
	  } else {
	    if(_id.roll()==1)
	      _cnr=12;
	    else
	      _cnr=13;
	  }
	} else if(_id.station()==4) {//Station 4
	  if (_id.sector()== 4) {	  
	    if ( _id.subsector()==1){
	      if(_id.roll()==1)
		_cnr=14;
	      else
		_cnr=15;
	    }else if (_id.subsector()==2){
	      if(_id.roll()==1)
		_cnr=16;
	      else
		_cnr=17;
	    }else  if ( _id.subsector()==3) {
	      if(_id.roll()==1)
		_cnr=18;
	      else
		_cnr=19;
	    }else if ( _id.subsector()==4){
	      if(_id.roll()==1)
		_cnr=20;
	      else
		_cnr=21;
	    }
	  } else {
	    if(_id.subsector()==1) {
	      if(_id.roll()==1)
		_cnr=14;
	      else 
		_cnr=15;
	    } else {
	      if(_id.roll()==1)
		_cnr=16;
	      else
		_cnr=17;
	    } 
	  } 
	}
      }else{//Endcap
	//	int seg=0;
	int nseg=36;
	int nsub=6;
	if ( _id.ring()==1 &&  _id.station() > 1) {
	  nsub=3;
	  nseg=18;
	}
	
	//	seg =(_id.sector()-1)*nsub + _id.subsector() ;
	  _cnr = (_id.subsector()-1)*3+_id.roll()+(_id.ring()-1)*nsub*3; 
      }
	return _cnr;
    }
    
  private:
      int _cnr;
  };
}

#endif
