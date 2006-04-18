#ifndef Geometry_RPCRollService_h
#define Geometry_RPCRollService_h
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
class StripTopology;
class RPCRollService{
 public:
  RPCRollService();
  RPCRollService(RPCRoll* roll);
  ~RPCRollService();
  int nstrips();
  LocalPoint  GlobalToLocalPoint(const GlobalPoint& gp);
  GlobalPoint LocalToGlobalPoint(const LocalPoint& lp);
  LocalPoint  CentreOfStrip(int strip);
  LocalPoint  CentreOfStrip(float strip);
  bool isBarrel();
  bool isForward();

 private:
  const StripTopology* topology();
 private:
  RPCRoll* roll_;
  const StripTopology* top_;
};
#endif
