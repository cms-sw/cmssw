#ifndef RPCGeometry_RPCGeomServ_h
#define RPCGeometry_RPCGeomServ_h

#include <string>
#include <vector>

class RPCDetId;
class RPCGeomServ{
 public:
  RPCGeomServ(const RPCDetId& id);
  virtual ~RPCGeomServ();
  virtual std::string shortname();
  virtual std::string name();
  virtual std::string chambername();
  virtual int eta_partition(); 
  virtual int chambernr();
  virtual int segment();
  virtual bool inverted();
  virtual bool zpositive();
  virtual bool aclockwise();
  std::vector<int> channelInChip();
 
protected:
  RPCGeomServ();
  
 protected:
  const RPCDetId* _id;
  std::string _n;
  std::string _sn;
  int _t;
  int _cnr;
  bool _z;
  bool _a;

};
#endif
