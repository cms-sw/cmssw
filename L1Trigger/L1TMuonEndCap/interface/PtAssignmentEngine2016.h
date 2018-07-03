#ifndef L1TMuonEndCap_PtAssignmentEngine2016_h
#define L1TMuonEndCap_PtAssignmentEngine2016_h

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2016.h"

class PtAssignmentEngine2016: public PtAssignmentEngine {
public:
  explicit PtAssignmentEngine2016(): PtAssignmentEngine(){}
  ~PtAssignmentEngine2016() override {}

  const PtAssignmentEngineAux2016& aux() const;

  float scale_pt  (const float pt, const int mode = 15) const override;
  float unscale_pt(const float pt, const int mode = 15) const override;
  address_t calculate_address(const EMTFTrack& track) const override;
  float calculate_pt_xml(const address_t& address) const override;
  float calculate_pt_xml(const EMTFTrack& track) const override;

private:
};

#endif
