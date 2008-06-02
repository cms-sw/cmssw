#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <sstream>
#include <iomanip>

RPCGeomServ::RPCGeomServ::RPCGeomServ(const RPCDetId& id) : 
  _id(&id), _n(""), _t(-99), _z(true), _a(true)
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
	if (_id->station()<=2)
	  if (_id->layer()==1)
	    os<<"in";
	  else
	    os<<"out";
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
	      os <<"-+";
	    }
	    else if ( _id->subsector()==3){
	      os <<"+-";
	    }
	    else if ( _id->subsector()==4){
	      os <<"++";
	    }
	  }
	  else {
	    if (_id->subsector()==1)
	      os <<"-";
	    else
	      os <<"+";
	  }
	}
	


	os<<"_";
	os <<"S"<<std::setw(2)<<std::setfill('0')
	   <<_id->sector()<<std::setfill(' ');
	os<<"_";
	buf += os.str();
      }
      {
	std::stringstream os;
	if (_id->roll()==1)
	  os<<"Backward";
	else if (_id->roll() == 3)
	  os<<"Forward";
	else if (_id->roll() == 2)
	os <<"Middle";
	buf += os.str();
      }
    }
    else {
      buf="D";
      
      {
	std::stringstream os;
	os << std::setw(2)<<std::setfill('+')<<_id->station()*_id->region()
	   <<std::setfill(' ')<<"_";
	buf += os.str();    
      }
      
      {
	std::stringstream os;
	os <<"RE"<<_id->station()<<"_"<<_id->ring();
	os <<"S"<<std::setw(2)<<std::setfill('0')<<this->segment();
	  
	buf += os.str();
      } 

      {
	buf += "_";
	std::stringstream os;
	if (_id->roll()==1)
	  os<<"A";
	else if (_id->roll() == 2)
	  os<<"B";
	else if (_id->roll() == 3)
	  os <<"C";
	else if (_id->roll() == 4)
	  os <<"D";
	buf += os.str();
      }
    }
    _n=buf;
  }
  return _n;
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
      
      if(_id->roll()==1)
	_cnr = 5;
      if(_id->roll()==3)
	_cnr=6;
      if(_id->roll()==2)
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
      
      if ( _id->subsector()==1){
	
	if(_id->roll()==1)
	  _cnr=14;
	else
	  _cnr=15;
	
      }
      
      if ( _id->subsector()==2){
	
	if(_id->roll()==1)
	  _cnr=16;
	else
	  _cnr=17;
	
      }
      
      if ( _id->subsector()==3) {
	
	if(_id->roll()==1)
	  _cnr=18;
	else
	  _cnr=19;
      }
      
      if ( _id->subsector()==4){
	
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
  int nsub=6;
  if ( _id->ring()==1 &&  _id->station() > 1) 
    nsub=3;
  return _id->subsector()+nsub*(_id->sector()-1);
      
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





RPCGeomServ::RPCGeomServ() : _id(0), _n(""), _t (-99), _cnr(-99), _z(false), _a(false)
{} 


