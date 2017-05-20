#ifndef L1TMuonEndCap_PtAssignmentEngine2016_hh
#define L1TMuonEndCap_PtAssignmentEngine2016_hh

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2016.hh"

class PtAssignmentEngine2016: public PtAssignmentEngine {
public:
  explicit PtAssignmentEngine2016():PtAssignmentEngine(){}
  ~PtAssignmentEngine2016(){}

  const PtAssignmentEngineAux2016& aux() const;

  virtual address_t calculate_address(const EMTFTrack& track) const;
  virtual float calculate_pt_xml(const address_t& address);
  virtual float calculate_pt_xml(const EMTFTrack& track);

private:
};

#endif
