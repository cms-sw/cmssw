#ifndef L1Trigger_DTTriggerPhase2_ShowerCandidate_h
#define L1Trigger_DTTriggerPhase2_ShowerCandidate_h
#include <iostream>
#include <memory>
#include <map>

#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"

class ShowerCandidate {
public:
  ShowerCandidate();
  ShowerCandidate& operator=(const ShowerCandidate& other);

  // setter methods
  void rawId(int rawId) { rawId_ = rawId; }
  void setBX(int bx) { bx_ = bx; }
  void setNhits(int nhits) { nhits_ = nhits; }
  void setMinWire(int wmin) { wmin_ = wmin; }
  void setMaxWire(int wmax) { wmax_ = wmax; }
  void setAvgPos(float avgPos) { avgPos_ = avgPos; }
  void setAvgTime(float avgTime) { avgTime_ = avgTime; }
  void flag() { shower_flag_ = true; }

  // getters methods
  int getRawId() { return rawId_; }
  int getBX() { return bx_; }
  int getNhits() { return nhits_; }
  int getMinWire() { return wmin_; }
  int getMaxWire() { return wmax_; }
  float getAvgPos() { return avgPos_; }
  float getAvgTime() { return avgTime_; }
  std::vector<int>& getWiresProfile() { return wires_profile_; }
  std::vector<int>& getWiresConstituents() { return wires_constituents_; }
  std::vector<int>& getWiresLayerConstituents() { return wires_layer_constituents_; }
  std::vector<int>& getWiresTdcConstituents() { return wires_tdc_constituents_; }
  bool isFlagged() { return shower_flag_; }

  // Other methods
  void clear();

private:
  //------------------------------------------------------------------
  //---  ShowerCandidate's data
  //------------------------------------------------------------------
  int nhits_;
  int rawId_;
  int bx_;
  int wmin_;
  int wmax_;
  bool shower_flag_;
  float avgPos_;
  float avgTime_;
  std::vector<int> wires_profile_;
  std::vector<int> wires_constituents_;
  std::vector<int> wires_layer_constituents_;
  std::vector<int> wires_tdc_constituents_;
};

typedef std::vector<ShowerCandidate> ShowerCandidates;
typedef std::shared_ptr<ShowerCandidate> ShowerCandidatePtr;
typedef std::vector<ShowerCandidatePtr> ShowerCandidatePtrs;
typedef std::map<DTSuperLayerId, ShowerCandidatePtr> ShowerCandidateMap;
typedef std::map<DTSuperLayerId, ShowerCandidatePtrs> ShowerCandidatesMap;
#endif
