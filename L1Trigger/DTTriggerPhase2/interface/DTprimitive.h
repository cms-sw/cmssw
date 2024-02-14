#ifndef L1Trigger_DTTriggerPhase2_DTprimitive_h
#define L1Trigger_DTTriggerPhase2_DTprimitive_h

#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include <ostream>
#include <memory>
#include <vector>

class DTPrimitive {
public:
  DTPrimitive();
  DTPrimitive(std::shared_ptr<DTPrimitive>& ptr);
  DTPrimitive(DTPrimitive* ptr);
  virtual ~DTPrimitive();

  bool isValidTime();
  float wireHorizPos();

  void setTimeCorrection(int time) { timeCorrection_ = time; };
  void setTDCTimeStamp(int tstamp) { tdcTimeStamp_ = tstamp; };
  void setOrbit(int orb) { orbit_ = orb; }
  void setPayload(double hitTag, int idx) { this->payLoad_[idx] = hitTag; };
  void setChannelId(int channel) { channelId_ = channel; };
  void setLayerId(int layer) { layerId_ = layer; };
  void setCameraId(int camera) { cameraId_ = camera; };
  void setSuperLayerId(int lay) { superLayerId_ = lay; };
  void setLaterality(cmsdt::LATERAL_CASES lat) { laterality_ = lat; };

  const int timeCorrection() const { return timeCorrection_; };
  const int tdcTimeStamp() const { return tdcTimeStamp_; };
  const int orbit() const { return orbit_; };
  const int tdcTimeStampNoOffset() const { return tdcTimeStamp_ - timeCorrection_; };
  const double payLoad(int idx) const { return payLoad_[idx]; };
  const int channelId() const { return channelId_; };
  const int layerId() const { return layerId_; };
  const int cameraId() const { return cameraId_; };
  const int superLayerId() const { return superLayerId_; };
  const cmsdt::LATERAL_CASES laterality() const { return laterality_; };

  bool operator==(const DTPrimitive& dtp) {
    return (tdcTimeStamp() == dtp.tdcTimeStamp() && channelId() == dtp.channelId() && layerId() == dtp.layerId() &&
            cameraId() == dtp.cameraId() && cameraId() == dtp.cameraId() && superLayerId() == dtp.superLayerId());
  }

private:
  int cameraId_;                     // Chamber ID
  int superLayerId_;                 // SL ID
  int layerId_;                      // Layer ID
  int channelId_;                    // Wire number
  cmsdt::LATERAL_CASES laterality_;  // LEFT, RIGHT, NONE

  int timeCorrection_;
  int tdcTimeStamp_;
  int orbit_;
  double payLoad_[cmsdt::PAYLOAD_ENTRIES];
};

typedef std::vector<DTPrimitive> DTPrimitives;
typedef std::shared_ptr<DTPrimitive> DTPrimitivePtr;
typedef std::vector<DTPrimitivePtr> DTPrimitivePtrs;

#endif
