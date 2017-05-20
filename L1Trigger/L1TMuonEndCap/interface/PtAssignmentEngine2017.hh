#ifndef L1TMuonEndCap_PtAssignmentEngine2017_hh
#define L1TMuonEndCap_PtAssignmentEngine2017_hh

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.hh"

class PtAssignmentEngine2017: public PtAssignmentEngine {
public:
  explicit PtAssignmentEngine2017():PtAssignmentEngine(){}
  ~PtAssignmentEngine2017(){}

  const PtAssignmentEngineAux2017& aux() const;

  virtual address_t calculate_address(const EMTFTrack& track) const;
  virtual float calculate_pt_xml(const address_t& address);
  virtual float calculate_pt_xml(const EMTFTrack& track);

private:
};

#endif
