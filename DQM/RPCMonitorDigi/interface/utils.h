#ifndef rpcdqm_utils_H
#define rpcdqm_utils_H

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <vector>
#include <iomanip>
#include <string>


namespace rpcdqm{

  enum RPCMeFLag{OCCUPANCY = 1, CLUSTERSIZE = 2, MULTIPLICITY =3, BX=4};

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
	      _cnr = 5;//RB2inB
	    else if(_id.roll()==3)
	      _cnr=6;//RB2inF
	    else if(_id.roll()==2)
	      _cnr=7;//RB2inM
	  }else{
	    if(_id.roll()==2)
	      _cnr=7;//RB2outM
	    else if(_id.roll()==1)
	      _cnr=8;//RB2outB
	    else if(_id.roll()==3)
	      _cnr=9;//RB2outF 
	  }
	}else  if(_id.station()==3){//Station 3
	  if(_id.subsector()==1){
	    if(_id.roll()==1)
	      _cnr=10;//RB3-B
	    else 
	      _cnr=11;//RB3-F
	  } else {
	    if(_id.roll()==1)
	      _cnr=12;//RB3+B
	    else
	      _cnr=13;//RB3+F
	  }
	} else if(_id.station()==4) {//Station 4
	  if (_id.sector()== 4) {	  
	    if ( _id.subsector()==2){//RB4--
	      if(_id.roll()==1)
		_cnr=14;
	      else
		_cnr=15;
	    }else if (_id.subsector()==3){//RB4-
	      if(_id.roll()==1)
		_cnr=16;
	      else
		_cnr=17;
	    }else  if ( _id.subsector()==1) {//RB4+
	      if(_id.roll()==1)
		_cnr=18;
	      else
		_cnr=19;
	    }else if ( _id.subsector()==4){//RB4++
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
	int nsub=6;
	if ( _id.ring()==1 &&  _id.station() > 1) {
	  nsub=3;
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


    void dolabeling (bool useRollInfo) {

      if(useRollInfo){
      ylabel[1] = "RB1in_B";
      ylabel[2] = "RB1in_F";
      ylabel[3] = "RB1out_B";
      ylabel[4] = "RB1out_F";
      ylabel[5] = "RB2in_B";
      ylabel[6] = "RB2in_F";
      
      ylabel[7] = "RB2in_M";
      ylabel[0] = "RB2out_M";
      
      ylabel[8] = "RB2out_B";
      ylabel[9] = "RB2out_F";
      ylabel[10] = "RB3-_B";
      ylabel[11] = "RB3-_F";
      ylabel[12] = "RB3+_B";
      ylabel[13] = "RB3+_F";
      ylabel[14] = "RB4,-_B";
      ylabel[15] = "RB4,-_F";
      ylabel[16] = "RB4+_B";
      ylabel[17] = "RB4+_F";
      ylabel[18] = "RB4--_B";
      ylabel[19] = "RB4--_F";
      ylabel[20] = "RB4++_B";
      ylabel[21] = "RB4++_F";
      }else{
	ylabel[1] = "RB1in";
	ylabel[2] = "";
	ylabel[3] = "RB1out";
	ylabel[4] = "";
	ylabel[5] = "RB2in";
	ylabel[6] = "";
	
	ylabel[7] = "";
	ylabel[0] = "";
	
	ylabel[8] = "RB2out";
	ylabel[9] = "";
	ylabel[10] = "RB3-";
	ylabel[11] = "";
	ylabel[12] = "RB3+";
	ylabel[13] = "";
	ylabel[14] = "RB4,-";
	ylabel[15] = "";
	ylabel[16] = "RB4+";
	ylabel[17] = "";
	ylabel[18] = "RB4--";
	ylabel[19] = "";
	ylabel[20] = "RB4++";
	ylabel[21] = "";
      }





    }


    void doEndcapLabeling(bool useRollInfo){          

      std::string rolls[3];
      rolls[0]="A";
      rolls[1]="B";
      rolls[2]="C";

      endcapYLabels_.clear();
      std::stringstream myLabel;

      for(int ring = 1 ; ring <=3; ring ++){
	for(int ch = 1; ch<=6; ch++){
	  for(int r=0; r<3; r++){
	    myLabel.str("");
	    myLabel<<"R"<<ring<<"_C"<<ch<<"_"<<rolls[r];
	    endcapYLabels_.push_back(myLabel.str());
	  }
	}
      }
    }

    std::string YLabel(int i) {

      return ylabel[i];
      
    }

    //use only with RollvsSector MEs
    void labelXAxisSector(MonitorElement * myMe){
      //before do some checks
      if (!myMe) return;

      std::stringstream xLabel;

      for(int x = 1; x<= myMe->getNbinsX(); x++){
	xLabel.str("");
	xLabel<<"Sec"<<x;
	myMe->setBinLabel(x, xLabel.str(), 1);
      }
    }

    //use only with RingvsSegment MEs
    void labelXAxisSegment(MonitorElement * myMe){
      //before do some checks
      if (!myMe) return;

      std::stringstream xLabel;

      myMe->setAxisTitle("Segments", 1);

      for(int x = 1; x<= myMe->getNbinsX(); x++){
	xLabel.str("");
	xLabel<<x;
	myMe->setBinLabel(x, xLabel.str(), 1);
      }
      
    }


    //use only with RollvsSector MEs
    void labelYAxisRoll(MonitorElement * myMe, int region, int ring, bool useRollInfo){
  
      //before do some checks
      if (!myMe) return;
  
      //set bin labels
      if(region == 0){
  
	//initialize label vector
	this->dolabeling(useRollInfo);  
	if(ring == -2 || ring == 2) ylabel[7]=ylabel[0];
  
	for(int y = 1; y<= myMe->getNbinsY() && y<22; y++)	  
	  myMe->setBinLabel(y, ylabel[y], 2);
	
      }else{//Endcap
   
	this->doEndcapLabeling(useRollInfo);
   
  	for(int y = 1; y<= myMe->getNbinsY() && y<(int)endcapYLabels_.size(); y++)	  
	  myMe->setBinLabel(y, endcapYLabels_[y], 2);
   
      }
    }


  //use only with RingvsSegment MEs
    void labelYAxisRing(MonitorElement * myMe, int numberOfRings, bool useRollInfo){
  
      //before do some checks
      if (!myMe) return;
      std::string labels[9];
      if(useRollInfo){
	labels[0]= "C";
	labels[1]= "Ring1 B";
	labels[2]= "A";
	labels[3]= "C";
	labels[4]= "Ring2 B";
	labels[5]= "A";
	labels[6]= "C";
	labels[7]= "Ring3 B";
	labels[8]= "A";
      }else{
	labels[0]= "";
	labels[1]= "Ring1";
	labels[2]= "";
	labels[3]= "";
	labels[4]= "Ring2";
	labels[5]= "";
	labels[6]= "";
	labels[7]= "Ring3";
	labels[8]= "";
	
      }
	int startBin ;
	(numberOfRings == 2 ? startBin = 3: startBin = 0);
	
	//set bin labels
	for(int y =1 ;y<= myMe->getNbinsY() && y<=9; y++ ){
	  myMe->setBinLabel(y,labels[y-1+startBin],2);
	  
	}
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
      std::vector<std::string>  endcapYLabels_;
      std::string ChLabel;
      std::vector<int> Wvector2;
      std::vector<int> Wvector1;
      std::string ylabel[22];
  };
}

#endif
