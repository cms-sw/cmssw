#ifndef L1TMuonEndCap_PtAssignmentEngine2017_h
#define L1TMuonEndCap_PtAssignmentEngine2017_h

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.h"

class PtAssignmentEngine2017: public PtAssignmentEngine {
public:
  explicit PtAssignmentEngine2017(): PtAssignmentEngine(){}
  ~PtAssignmentEngine2017(){}

  const PtAssignmentEngineAux2017& aux() const;

  virtual float scale_pt  (const float pt, const int mode = 15) const;
  virtual float unscale_pt(const float pt, const int mode = 15) const;
  virtual address_t calculate_address(const EMTFTrack& track) const;
  virtual float calculate_pt_xml(const address_t& address) const;
  virtual float calculate_pt_xml(const EMTFTrack& track) const;

private:
  int version_;
};

#endif
