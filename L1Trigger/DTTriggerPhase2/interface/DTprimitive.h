#ifndef L1Trigger_DTTriggerPhase2_DTprimitive_h
#define L1Trigger_DTTriggerPhase2_DTprimitive_h

#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include <ostream>
#include <memory>
#include <vector>

using namespace cmsdt;

class DTPrimitive {
public:
  DTPrimitive();
  DTPrimitive(std::shared_ptr<DTPrimitive>& ptr);
  DTPrimitive(DTPrimitive* ptr);
  virtual ~DTPrimitive();

  /* Este método se implementará en la FPGA mediante la comprobación de un
       bit que indique la validez del valor. En el software lo hacemos
       representando como valor no válido, un número negativo cualquiera */
  bool isValidTime(void);
  float wireHorizPos(void);

  void setTimeCorrection(int time) { timeCorrection_ = time; };
  void setTDCTimeStamp(int tstamp) { tdcTimeStamp_ = tstamp; };
  void setOrbit(int orb) { orbit_ = orb; }
  void setPayload(double hitTag, int idx) { this->payLoad_[idx] = hitTag; };
  void setChannelId(int channel) { channelId_ = channel; };
  void setLayerId(int layer) { layerId_ = layer; };
  void setCameraId(int camera) { cameraId_ = camera; };
  void setSuperLayerId(int lay) { superLayerId_ = lay; };
  void setLaterality(LATERAL_CASES lat) { laterality_ = lat; };

  const int timeCorrection(void) { return timeCorrection_; };
  const int tdcTimeStamp(void) { return tdcTimeStamp_; };
  const int orbit(void) { return orbit_; };
  const int tdcTimeStampNoOffset(void) { return tdcTimeStamp_ - timeCorrection_; };
  const double payLoad(int idx) { return payLoad_[idx]; };
  const int channelId(void) { return channelId_; };
  const int layerId(void) { return layerId_; };
  const int cameraId(void) { return cameraId_; };
  const int superLayerId(void) { return superLayerId_; };
  const LATERAL_CASES laterality(void) { return laterality_; };

private:
  int cameraId_;              // Chamber ID
  int superLayerId_;          // SL ID
  int layerId_;               // Layer ID
  int channelId_;             // Wire number
  LATERAL_CASES laterality_;  // LEFT, RIGHT, NONE

  int timeCorrection_;  // Correccion temporal por electronica, etc...
  int tdcTimeStamp_;    // Tiempo medido por el TDC
  int orbit_;           // Número de órbita
  double payLoad_[PAYLOAD_ENTRIES];
};

typedef std::vector<DTPrimitive> DTPrimitives;
typedef std::shared_ptr<DTPrimitive> DTPrimitivePtr;
typedef std::vector<DTPrimitivePtr> DTPrimitivePtrs;

#endif
