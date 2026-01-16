#ifndef L1TMuonEndCap_PtAssignmentEngineDxy_h
#define L1TMuonEndCap_PtAssignmentEngineDxy_h

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <array>

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

class PtAssignmentEngineDxy {
public:
  explicit PtAssignmentEngineDxy(std::shared_ptr<hls4mlEmulator::Model> model);
  virtual ~PtAssignmentEngineDxy();

  void configure(int verbose);

  const PtAssignmentEngineAux2017& aux() const;

  virtual void calculate_pt_dxy(const EMTFTrack& track, emtf::Feature& feature, emtf::Prediction& prediction) const;

  virtual void preprocessing_dxy(const EMTFTrack& track, emtf::Feature& feature) const;

  virtual void call_hls_dxy(const emtf::Feature& feature, emtf::Prediction& prediction) const;

protected:
  int verbose_;
  std::shared_ptr<hls4mlEmulator::Model> model_;
};

#endif
