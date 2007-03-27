#ifndef RPCGeometry_RPCGeomServ_h
#define RPCGeometry_RPCGeomServ_h

#include <string>

class RPCDetId;
class RPCGeomServ{
 public:
  RPCGeomServ(RPCDetId* id);
  virtual ~RPCGeomServ();
  virtual std::string name();
  virtual int eta_partition(); 
  virtual bool inverted();
 protected:
  RPCGeomServ();
 protected:
  RPCDetId* _id;
  std::string _n;
  int _t;
  bool _i;

};
#endif
