#ifndef __L1Trigger_L1THGCal_HGCalConcentratorAutoEncoderImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorAutoEncoderImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalConcentratorData.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include <vector>

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class HGCalConcentratorAutoEncoderImpl {
public:
  HGCalConcentratorAutoEncoderImpl(const edm::ParameterSet& conf);

  void select(unsigned nLinks,
              const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
              std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput,
              std::vector<l1t::HGCalConcentratorData>& ae_EncodedOutput);

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }

private:
  static constexpr int nTriggerCells_ = 48;

  static constexpr int maxAEInputSize_ = 192;

  static constexpr int nEncodedLayerNodes_ = 16;

  static constexpr unsigned int maxNumberOfLinks_ = 13;

  static constexpr int cellUVSize_ = 8;

  static constexpr int encoderTensorDims_ = 4;

  static constexpr int decoderTensorDims_ = 2;

  static constexpr int cellUVremap_[cellUVSize_][cellUVSize_] = {{47, 46, 45, 44, -1, -1, -1, -1},
                                                                 {16, 43, 42, 41, 40, -1, -1, -1},
                                                                 {20, 17, 39, 38, 37, 36, -1, -1},
                                                                 {24, 21, 18, 35, 34, 33, 32, -1},
                                                                 {28, 25, 22, 19, 3, 7, 11, 15},
                                                                 {-1, 29, 26, 23, 2, 6, 10, 14},
                                                                 {-1, -1, 30, 27, 1, 5, 9, 13},
                                                                 {-1, -1, -1, 31, 0, 4, 8, 12}};
  static constexpr int ae_outputCellU_[nTriggerCells_] = {7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4,
                                                          1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7,
                                                          3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0};
  static constexpr int ae_outputCellV_[nTriggerCells_] = {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                                                          0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                                          6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0};

  unsigned int nInputs_;
  std::vector<int> cellRemap_;
  std::vector<int> cellRemapNoDuplicates_;
  std::vector<uint> encoderShape_;
  std::vector<uint> decoderShape_;
  int bitsPerInput_;
  int maxBitsPerOutput_;
  std::vector<int> outputBitsPerLink_;

  std::vector<edm::ParameterSet> modelFilePaths_;

  std::string inputTensorName_encoder_;
  std::string outputTensorName_encoder_;
  std::unique_ptr<tensorflow::GraphDef> graphDef_encoder_;
  std::vector<std::unique_ptr<tensorflow::Session>> session_encoder_;

  std::string inputTensorName_decoder_;
  std::string outputTensorName_decoder_;
  std::unique_ptr<tensorflow::GraphDef> graphDef_decoder_;
  std::vector<std::unique_ptr<tensorflow::Session>> session_decoder_;

  std::vector<unsigned int> linkToGraphMap_;

  double zeroSuppresionThreshold_;
  bool bitShiftNormalization_;
  bool saveEncodedValues_;
  bool preserveModuleSum_;

  std::array<double, nEncodedLayerNodes_> ae_encodedLayer_;

  HGCalTriggerTools triggerTools_;
};

#endif
