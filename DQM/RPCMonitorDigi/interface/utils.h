#ifndef rpcdqm_utils_H
#define rpcdqm_utils_H

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <vector>
#include <iomanip>
#include <string>
using namespace std;

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
	int nseg=36;
	int nsub=6;
	if ( _id.ring()==1 &&  _id.station() > 1) {
	  nsub=3;
	  nseg=18;
	}
	  _cnr = (_id.subsector()-1)*3+_id.roll()+(_id.ring()-1)*nsub*3; 
      }
	return _cnr;
    }

    void fillvect() {
      Wvector2.push_back(0);    //Sec 0 - doen't exist ;)
      Wvector2.push_back(1140); //Sec1
      Wvector2.push_back(1140); //Sec2
      Wvector2.push_back(1140); //Sec3
      Wvector2.push_back(1236); //Sec4
      Wvector2.push_back(1140); //Sec5
      Wvector2.push_back(1140); //Sec6
      Wvector2.push_back(1140); //Sec7
      Wvector2.push_back(1116); //Sec8
      Wvector2.push_back(1044); //Sec9
      Wvector2.push_back(1188); //Sec10
      Wvector2.push_back(1044); //Sec11
      Wvector2.push_back(1166); //Sec12

      Wvector1.push_back(0);       //Sec 0 - doen't exist ;) 
      Wvector1.push_back(1146); //Sec1
      Wvector1.push_back(1146); //Sec2
      Wvector1.push_back(1146); //Sec3
      Wvector1.push_back(1242); //Sec4
      Wvector1.push_back(1146); //Sec5
      Wvector1.push_back(1146); //Sec6
      Wvector1.push_back(1146); //Sec7
      Wvector1.push_back(1122); //Sec8
      Wvector1.push_back(1050); //Sec9
      Wvector1.push_back(1194); //Sec10
      Wvector1.push_back(1050); //Sec11
      Wvector1.push_back(1122); //Sec12
      
    }
    
    std::vector<int> SectorStrips2(){
      return Wvector2;
    }
    
    std::vector<int> SectorStrips1(){
      return Wvector1;
    }


    void dolabeling () {
      
      ylabel[1] = "RB1in_B";
      ylabel[2] = "RB1in_F";
      ylabel[3] = "RBiout_B";
      ylabel[4] = "RB1out_F";
      ylabel[5] = "RB2in_B";
      ylabel[6] = "RB2in_F";
      
      ylabel[7] = "RB2in_M";
      ylabel[0] = "RB1out_M";
      
      ylabel[8] = "RB2out_B";
      ylabel[9] = "RB2out_F";
      ylabel[10] = "RB3-_F";
      ylabel[11] = "RB3-_B";
      ylabel[12] = "RB3+_B";
      ylabel[13] = "RB3+_F";
      ylabel[14] = "RB4,-,--_B";
      ylabel[15] = "RB4,-,--,F";
      ylabel[16] = "RB4+,-+_B";
      ylabel[17] = "RB4+,-+_F";
      ylabel[18] = "RB4+-_B";
      ylabel[19] = "RB4+-_F";
      ylabel[20] = "RB4++_B";
      ylabel[21] = "RB4++_F";
           
    }

    string YLabel(int i) {

      return ylabel[i];
      
    }


  
    int detId2ChamberNr(const RPCDetId & _id){
      if(_id.region()==0){//Barrel
	if( _id.station() ==1) {// Station1
	  if(_id.layer() ==1) {
	   
	    ch=1; //RB1in
	    
	  } else {
	    ch=2; //RB1out
	  }

	}else if (_id.station()==2) {//Station 2	
	  if(_id.layer()==1) {
	    
	      ch=3;//RB2in
	  }else{
	    
	    ch=4; //RB2out
	  }
	}else  if(_id.station()==3){//Station 3
	  if(_id.subsector()==1){
	    ch=5; //RB3+
	  } else {

	    ch=6; //RB3-
	  }
	} else if(_id.station()==4) {//Station 4
	  if (_id.sector()== 4) {	  
	    if ( _id.subsector()==1){
	      
	      ch=7; //RB4-
	      
	    }else if (_id.subsector()==2){
	      ch=8; //RB4+
	    }else  if ( _id.subsector()==3) {
	      ch=9; //RB4--
	    }else if ( _id.subsector()==4){
	      ch=10; //RB4++
	    }
	  } else {
	    if(_id.subsector()==1) ch= 7; //RB4-
	    else ch= 8; //RB4+
	  } 
	}
      }else{//Endcap
      }
	return ch;
    }

 
  
     
std::string detId2ChamberLabel(const RPCDetId & _id){
      if(_id.region()==0){//Barrel
	if( _id.station() ==1) {// Station1
	  if(_id.layer() ==1) {
	   
	    ChLabel="RB1in";
	    
	  } else {
	    ChLabel="RB1out";
	  }

	}else if (_id.station()==2) {//Station 2	
	  if(_id.layer()==1) {
	    
	    ChLabel="RB2in";
	  }else{
	    
	    ChLabel="RB2out";
	  }
	}else  if(_id.station()==3){//Station 3
	  if(_id.subsector()==1){
	    ChLabel="RB3+";
	  } else {

	    ChLabel="RB3-";
	  }
	} else if(_id.station()==4) {//Station 4
	  if (_id.sector()== 4) {	  
	    if ( _id.subsector()==1){
	      
	      ChLabel="RB4-";
	      
	    }else if (_id.subsector()==2){
	      ChLabel="RB4+";
	    }else  if ( _id.subsector()==3) {
	      ChLabel="RB4--";
	    }else if ( _id.subsector()==4){
	      ChLabel="RB4++";
	    }
	  } else {
	    if(_id.subsector()==1) ChLabel="RB4-";
	    else ChLabel="RB4-";
	  } 
	}
      }else{//Endcap
      }
	return ChLabel;
    }

 


    
  private:
      int _cnr;
      int ch;
      
      std::string ChLabel;
      std::vector<int> Wvector2;
      std::vector<int> Wvector1;
      string ylabel[22];
     
      

  };
}

#endif
