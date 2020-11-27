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
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"



class PtAssignmentEngineDxy {
public:
  explicit PtAssignmentEngineDxy(const std::string pbFileNameDxy);
  virtual ~PtAssignmentEngineDxy();

  void configure(
          int verbose
  );

  const PtAssignmentEngineAux2017& aux() const;

  virtual void calculate_pt_dxy(const EMTFTrack& track, emtf::Feature& feature, emtf::Prediction& prediction) const;

  virtual void preprocessing_dxy(const EMTFTrack& track, emtf::Feature& feature) const;

  virtual void call_tensorflow_dxy(const emtf::Feature& feature, emtf::Prediction& prediction) const;


protected:

  int verbose_;

  std::string cmssw_base_;
  tensorflow::GraphDef* graphDefDxy_;
  tensorflow::Session* sessionDxy_;
  std::string pbFileNameDxy_;
  std::string inputNameDxy_;
  std::vector<std::string> outputNamesDxy_;

};

#endif