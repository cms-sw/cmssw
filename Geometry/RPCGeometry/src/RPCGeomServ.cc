#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <sstream>
#include <iomanip>

RPCGeomServ::RPCGeomServ::RPCGeomServ(const RPCDetId& id) : 
  _id(&id), _n(""), _sn(""), _t(-99), _z(true), _a(true)
{}


RPCGeomServ::~RPCGeomServ()
{}
 
std::string 
RPCGeomServ::name()
{
  if (_n.size()<1){
    std::string buf;
    
    if (_id->region()==0){
      buf="W";
      {
	std::stringstream os;
	os << std::setw(2)<<std::setfill('+')<<_id->ring()
	   <<std::setfill(' ')<<"_";
	buf += os.str();
      }
      
      {
	std::stringstream os;
	os <<"RB"<<_id->station();
	if (_id->station()<=2) {
	  if (_id->layer()==1)
	    os<<"in";
	  else
	    os<<"out";
	}
	//os<<"_";
	buf += os.str();
      }
      
      
      {
	std::stringstream os;
	//	os <<"S"<<std::setw(2)<<std::setfill('0')
	//   <<_id->sector()<<std::setfill(' ');
	if (_id->station()>2){
	  if (_id->sector()== 4 && _id->station()==4){
	    if ( _id->subsector()==1){
	      os<<"--";
	    }
	    else if ( _id->subsector()==2){
	      os <<"-";
	    }
	    else if ( _id->subsector()==3){
	      os <<"+";
	    }
	    else if ( _id->subsector()==4){
	      os <<"++";
	    }
	  }
	  
	  if(_id->station()==3){
	    if (_id->subsector()==1)
	      os <<"-";
	    else
	      os <<"+";
	  }else if(_id->station()==4 && _id->sector()!=9 && _id->sector()!=11 && _id->sector()!=4){
	    if (_id->subsector()==1)
	      os <<"-";
	    else
	      os <<"+";
	  }
	}
	
	os<<"_";
	os <<"S"<<std::setw(2)<<std::setfill('0')
	   <<_id->sector()<<std::setfill(' ');
	buf += os.str();
      }
      {
	std::stringstream os;
	if (_id->roll()==1)
	  os<<"_Backward";
	else if (_id->roll() == 3)
	  os<<"_Forward";
	else if (_id->roll() == 2)
	os <<"_Middle";
	buf += os.str();
      }
    }
    else {
      buf="RE";
      
      {
	std::stringstream os;
	os << std::setw(2)<<std::setfill('+')<<_id->station()*_id->region()
	   <<std::setfill(' ')<<"_";
	buf += os.str();    
      }
      
      {
	std::stringstream os;
	os <<"R"<<_id->ring();
	os <<"_CH"<<std::setw(2)<<std::setfill('0')<<this->segment();
	buf += os.str();
      } 

      {
	std::stringstream os;
	if (_id->roll()==1)
	  os<<"_A";
	else if (_id->roll() == 2)
	  os<<"_B";
	else if (_id->roll() == 3)
	  os <<"_C";
	else if (_id->roll() == 4)
	  os <<"_D";
	buf += os.str();
      }
    }
    _n=buf;
  }
  return _n;
}
 

std::string 
RPCGeomServ::chambername()
{
  if (_n.size()<1){
    std::string buf;
    
    if (_id->region()==0){
      buf="W";
      {
	std::stringstream os;
	os << std::setw(2)<<std::setfill('+')<<_id->ring()
	   <<std::setfill(' ')<<"_";
	buf += os.str();
      }
      
      {
	std::stringstream os;
	os <<"RB"<<_id->station();
	if (_id->station()<=2) {
	  if (_id->layer()==1)
	    os<<"in";
	  else
	    os<<"out";
	}
	//os<<"_";
	buf += os.str();
      }
      
      
      {
	std::stringstream os;
	//	os <<"S"<<std::setw(2)<<std::setfill('0')
	//   <<_id->sector()<<std::setfill(' ');
	if (_id->station()>2){
	  if (_id->sector()== 4 && _id->station()==4){
	    if ( _id->subsector()==1){
	      os<<"--";
	    }
	    else if ( _id->subsector()==2){
	      os <<"-";
	    }
	    else if ( _id->subsector()==3){
	      os <<"+";
	    }
	    else if ( _id->subsector()==4){
	      os <<"++";
	    }
	  }
	  
	  if(_id->station()==3){
	    if (_id->subsector()==1)
	      os <<"-";
	    else
	      os <<"+";
	  }else if(_id->station()==4 && _id->sector()!=9 && _id->sector()!=11 && _id->sector()!=4){
	    if (_id->subsector()==1)
	      os <<"-";
	    else
	      os <<"+";
	  }
	}
	
	os<<"_";
	os <<"S"<<std::setw(2)<<std::setfill('0')
	   <<_id->sector()<<std::setfill(' ');
	buf += os.str();
      }
      
    }
    else {
      buf="RE";
      
      {
	std::stringstream os;
	os << std::setw(2)<<std::setfill('+')<<_id->station()*_id->region()
	   <<std::setfill(' ')<<"_";
	buf += os.str();    
      }
      
      {
	std::stringstream os;
	os <<"R"<<_id->ring();
	os <<"_CH"<<std::setw(2)<<std::setfill('0')<<this->segment();
	buf += os.str();
      } 

      
    }
    _n=buf;
  }
  return _n;
}

std::string 
RPCGeomServ::shortname()
{
  if (_sn.size()<1)
    {
    std::string buf;

    if (_id->region()==0){
      std::stringstream os;
      os <<"RB"<<_id->station();
      if (_id->station()<=2){
	if (_id->layer()==1){
	  os<<"in";
	}else{
	  os<<"out";
	}
      }else{
	if (_id->sector()== 4 && _id->station()==4){
	  if ( _id->subsector()==1){
	    os<<"--";
	  }
	  else if ( _id->subsector()==2){
	    os <<",-";
	  }
	  else if ( _id->subsector()==3){
	    os <<"+";
	  }
	  else if ( _id->subsector()==4){
	    os <<"++";
	  }
	}else{
	  if (_id->subsector()==1)
	    os <<",-";
	  else
	    os <<"+";
	}
      }
      if (_id->roll()==1)
	os<<" B";
      else if (_id->roll() == 3)
	os<<" F";
      else if (_id->roll() == 2)
	os<<" M";
      buf += os.str();
    }
    else {
      std::stringstream os;
      os <<"Ri"<<_id->ring()<<" Su"<<_id->subsector();
      buf += os.str();
    }
    _sn=buf;
  }
  return _sn;
}

//returns a vector with number of channels for each chip in each FEB
std::vector<int> RPCGeomServ::channelInChip(){

  std::vector<int> chipCh(4,8);//Endcap
  
  if(_id->region()==0){//Barrel
    chipCh.clear();

    if (_id->station()<3 && _id->layer()==1){ // i.e. RB1in ||RB2in  
      chipCh.push_back(7);
      chipCh.push_back(8);
    }else if (_id->station() == 1 || _id->station() == 3){//i.e. RB1out || RB3 
      chipCh.push_back(7);
      chipCh.push_back(7);
    }else if (_id->station() == 2){// i.e. RB2out
      chipCh.push_back(6);
      chipCh.push_back(8);
    }else if (_id->sector() == 4 || _id->sector()==10 ||(_id->sector() == 8 &&  _id->subsector()!=1) || (_id->sector() == 12  &&  _id->subsector()==1)){
      chipCh.push_back(6);//i.e. Sector 4 &  10 RB4 and Sector 8 &12 RB4+
      chipCh.push_back(6);
    }else {
      chipCh.push_back(8);
      chipCh.push_back(8);
    }	
  }

  return chipCh;
}


int 
RPCGeomServ::eta_partition()
{
  if (_t<-90){
    if (_id->region() == 0 ){
      if (this->inverted()) {
	_t = 3*(_id->ring())+ (3-_id->roll())-1;
      }else{
	_t = 3*(_id->ring())+ _id->roll()-2;
      }
    }else{
      _t = _id->region() * (3*(3-_id->ring()) + _id->roll() + 7);
    }
  }
  return _t;
} 

int
RPCGeomServ::chambernr()
{

  // Station1
  if( _id->station() ==1) {
    
    // in
    if(_id->layer() ==1) { 
      
      if(_id->roll()==1) 
	_cnr = 1;
      else 
	_cnr = 2;
    }
    //out
    else 
      {
	if(_id->roll()==1) 
	  _cnr = 3;
	else 
	  _cnr = 4;
	
      }
  }


  //Station 2
  if (_id->station()==2) {
   
    //in
    if(_id->layer()==1) {
      
      if(_id->roll()==1)//backward
	_cnr = 5;
      if(_id->roll()==3)//forward
	_cnr=6;
      if(_id->roll()==2)//middle
	_cnr=7;
    }
    //out
    else {
      
      if(_id->roll()==2)
	
	_cnr=7;

      if(_id->roll()==1)
	_cnr=8;
      if(_id->roll()==3)
	_cnr=9;
    
    }
  }
    
  //RB3- RB3+
  if(_id->station()==3)
    {
      if(_id->subsector()==1) {
	
	if(_id->roll()==1)
	  _cnr=10;
	else 
	  _cnr=11;
      }
      else {
	
	if(_id->roll()==1)
	  _cnr=12;
	else
	  _cnr=13;
      }
      
    }

  //RB4
  if(_id->station()==4) {
    
    if (_id->sector()== 4) {
      
      if ( _id->subsector()==2){//RB4-
	
	if(_id->roll()==1)
	  _cnr=14;
	else
	  _cnr=15;
	
      }
      
      if ( _id->subsector()==3){//RB4+
	
	if(_id->roll()==1)
	  _cnr=16;
	else
	  _cnr=17;
	
      }
      
      if ( _id->subsector()==1) {//RB4--
	
	if(_id->roll()==1)
	  _cnr=18;
	else
	  _cnr=19;
      }
      
      if ( _id->subsector()==4){//RB4++
	
	if(_id->roll()==1)
	  _cnr=20;
	else
	  _cnr=21;
	
      }
      
    }  
    
    else 
      
      {
	if(_id->subsector()==1) {
	  
	  if(_id->roll()==1)
	    _cnr=14;
	  else 
	    _cnr=15;
	}
	else {
	  
	  if(_id->roll()==1)
	    _cnr=16;
	  else
	    _cnr=17;
	} 
      } 
  }
  

  // _cnr=10;
  return _cnr;
  
}

int 
RPCGeomServ::segment(){
  int seg=0;
  int nsec=36;
  int nsub=6;
  if ( _id->ring()==1 &&  _id->station() > 1) {
    nsub=3;
    nsec=18;
  }
  seg =_id->subsector()+nsub*(_id->sector()-1);//+1;
  //  if(seg==nsec+1)seg=1;
  return seg;
}

bool
RPCGeomServ::inverted()
{
  // return !(this->zpositive() && this->aclockwise());
  return !(this->zpositive());
}


bool
RPCGeomServ::zpositive()
{
  if (_id->region()==0 && _t<-90 ){
    if (_id->ring()<0){
      _z=false;
    }
    if (_id->ring()==0){
      if (_id->sector() == 1 || _id->sector() == 4 ||
	  _id->sector() == 5 || _id->sector() == 8 ||
	  _id->sector() == 9 || _id->sector() == 12){
	_z=false;
      }
    } 
  }
 
  return _z;
}

bool
RPCGeomServ::aclockwise()
{
  if (_id->region()==0 && _t<-90 ){
    if (_id->ring() > 0){
      if (_id->layer()==2){
	_a=false;
      }
    }else if(_id->ring() <0){
      if (_id->layer()==1){
	_a=false;
      }
    }else if(_id->ring() ==0) {
      if ((_id->sector() == 1 || _id->sector() == 4 ||
	   _id->sector() == 5 || _id->sector() == 8 ||
	   _id->sector() == 9 || _id->sector() == 12) && _id->layer()==1)
	_a=false;
      else if ((_id->sector() == 2 || _id->sector() == 3 ||
		_id->sector() == 6 || _id->sector() == 7 ||
		_id->sector() == 10|| _id->sector() == 11) && _id->layer()==2)
	_a=false;
    }
  }
  return _a;
}





RPCGeomServ::RPCGeomServ() : _id(0), _n(""), _sn(""), _t (-99), _z(false), _a(false)
{} 


