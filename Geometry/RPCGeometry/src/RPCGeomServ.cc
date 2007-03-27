#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"


RPCGeomServ::RPCGeomServ::RPCGeomServ(RPCDetId* id) : 
  _id(id), _n(""), _t(-99), _i(false)
{}


RPCGeomServ::~RPCGeomServ()
{}
  

std::string 
RPCGeomServ::name()
{
  if (_n.size()<1){

  }
  return _n;
}
 
int 
RPCGeomServ::eta_partition()
{
  if (_t<-90){
    if (_id->region() == 0 ){
      if (this->inverted()) {
	_t = 3*(_id->ring())+ (4-_id->roll())-1;
      }else{
	_t = 3*(_id->ring())+ _id->roll()-1;
      }
    }else{
      _t = 4*(_id->ring());
    }
  }
  return _t;
} 
 

RPCGeomServ::RPCGeomServ() : _id(0), _n(""), _t (-99), _i(false)
{} 


bool
RPCGeomServ::inverted()
{

  if (_t<-90){
    // do the calculation
  }
  return _i;
}
