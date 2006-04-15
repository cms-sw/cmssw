#ifndef Geometry_RPCRollService_h
#define Geometry_RPCRollService_h
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

class RPCRollService{
 public:
  RPCRollService();
  RPCRollService(RPCRoll* roll);
  ~RPCRollService();
  int nstrips();
  GlobalPoint GlobalToLocalPoint(const LocalPoint& lp);
  LocalPoint  LocalToGlobalPoint(const GlobalPoint& gp);
  bool isBarrel();
  bool isForward();

 private:
  RPCRoll* roll_;
};
#endif
