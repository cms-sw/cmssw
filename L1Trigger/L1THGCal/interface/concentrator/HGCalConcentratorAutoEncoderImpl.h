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

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

private:
  static constexpr int nTriggerCells_ = 48;
  static constexpr int nEncodedLayerNodes_ = 16;

  static constexpr int cellRemapRowOffset_ = 8;

  static constexpr unsigned int encoderShape_0_ = 1;
  static constexpr unsigned int encoderShape_1_ = 4;
  static constexpr unsigned int encoderShape_2_ = 4;
  static constexpr unsigned int encoderShape_3_ = 3;

  static constexpr unsigned int decoderShape_0_ = 1;
  static constexpr unsigned int decoderShape_1_ = 16;

  static constexpr unsigned int maxNumberOfLinks_ = 13;

  std::vector<int> cellRemap_;
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
  bool saveEncodedValues_;
  bool preserveModuleSum_;

  std::array<int, nTriggerCells_> ae_outputCellU_;
  std::array<int, nTriggerCells_> ae_outputCellV_;

  std::array<double, nEncodedLayerNodes_> ae_encodedLayer_;

  HGCalTriggerTools triggerTools_;
};

#endif
