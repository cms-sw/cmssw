#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFEProcessorSums.h"

DEFINE_EDM_PLUGIN(HGCalVFEProcessorBaseFactory, HGCalVFEProcessorSums, "HGCalVFEProcessorSums");

HGCalVFEProcessorSums::HGCalVFEProcessorSums(const edm::ParameterSet& conf) : HGCalVFEProcessorBase(conf) {
  vfeLinearizationSiImpl_ =
      std::make_unique<HGCalVFELinearizationImpl>(conf.getParameter<edm::ParameterSet>("linearizationCfg_si"));
  vfeLinearizationScImpl_ =
      std::make_unique<HGCalVFELinearizationImpl>(conf.getParameter<edm::ParameterSet>("linearizationCfg_sc"));

  vfeSummationImpl_ = std::make_unique<HGCalVFESummationImpl>(conf.getParameter<edm::ParameterSet>("summationCfg"));

  vfeCompressionLDMImpl_ =
      std::make_unique<HGCalVFECompressionImpl>(conf.getParameter<edm::ParameterSet>("compressionCfg_ldm"));
  vfeCompressionHDMImpl_ =
      std::make_unique<HGCalVFECompressionImpl>(conf.getParameter<edm::ParameterSet>("compressionCfg_hdm"));

  calibrationEE_ =
      std::make_unique<HGCalTriggerCellCalibration>(conf.getParameter<edm::ParameterSet>("calibrationCfg_ee"));
  calibrationHEsi_ =
      std::make_unique<HGCalTriggerCellCalibration>(conf.getParameter<edm::ParameterSet>("calibrationCfg_hesi"));
  calibrationHEsc_ =
      std::make_unique<HGCalTriggerCellCalibration>(conf.getParameter<edm::ParameterSet>("calibrationCfg_hesc"));
  calibrationNose_ =
      std::make_unique<HGCalTriggerCellCalibration>(conf.getParameter<edm::ParameterSet>("calibrationCfg_nose"));
}

void HGCalVFEProcessorSums::run(const HGCalDigiCollection& digiColl,
                                l1t::HGCalTriggerCellBxCollection& triggerCellColl) {
  vfeSummationImpl_->setGeometry(geometry());
  calibrationEE_->setGeometry(geometry());
  calibrationHEsi_->setGeometry(geometry());
  calibrationHEsc_->setGeometry(geometry());
  calibrationNose_->setGeometry(geometry());
  triggerTools_.setGeometry(geometry());

  std::vector<HGCalDataFrame> dataframes;
  std::vector<std::pair<DetId, uint32_t>> linearized_dataframes;
  std::unordered_map<uint32_t, uint32_t> tc_payload;
  std::unordered_map<uint32_t, std::array<uint64_t, 2>> tc_compressed_payload;

  // Remove disconnected modules and invalid cells
  for (const auto& digiData : digiColl) {
    if (!geometry()->validCell(digiData.id()))
      continue;
    uint32_t module = geometry()->getModuleFromCell(digiData.id());

    // no disconnected layer for HFNose
    if (DetId(digiData.id()).subdetId() != ForwardSubdetector::HFNose) {
      if (geometry()->disconnectedModule(module))
        continue;
    }

    dataframes.emplace_back(digiData.id());
    for (int i = 0; i < digiData.size(); i++) {
      dataframes.back().setSample(i, digiData.sample(i));
    }
  }
  if (dataframes.empty())
    return;

  constexpr int kHighDensityThickness = 0;
  bool isSilicon = triggerTools_.isSilicon(dataframes[0].id());
  bool isEM = triggerTools_.isEm(dataframes[0].id());
  bool isNose = triggerTools_.isNose(dataframes[0].id());
  int thickness = triggerTools_.thicknessIndex(dataframes[0].id());
  // Linearization of ADC and TOT values to the same LSB
  if (isSilicon) {
    vfeLinearizationSiImpl_->linearize(dataframes, linearized_dataframes);
  } else {
    vfeLinearizationScImpl_->linearize(dataframes, linearized_dataframes);
  }
  // Sum of sensor cells into trigger cells
  vfeSummationImpl_->triggerCellSums(linearized_dataframes, tc_payload);
  // Compression of trigger cell charges to a floating point format
  if (thickness == kHighDensityThickness) {
    vfeCompressionHDMImpl_->compress(tc_payload, tc_compressed_payload);
  } else {
    vfeCompressionLDMImpl_->compress(tc_payload, tc_compressed_payload);
  }

  // Transform map to trigger cell vector
  for (const auto& [tc_id, tc_value] : tc_payload) {
    if (tc_value > 0) {
      const auto& [tc_compressed_code, tc_compressed_value] = tc_compressed_payload[tc_id];

      if (tc_compressed_value > std::numeric_limits<int>::max())
        edm::LogWarning("CompressedValueDowncasting") << "Compressed value cannot fit into 32-bit word. Downcasting.";

      l1t::HGCalTriggerCell triggerCell(
          reco::LeafCandidate::LorentzVector(), static_cast<int>(tc_compressed_value), 0, 0, 0, tc_id);

      if (tc_compressed_code > std::numeric_limits<uint32_t>::max())
        edm::LogWarning("CompressedValueDowncasting") << "Compressed code cannot fit into 32-bit word. Downcasting.";

      triggerCell.setCompressedCharge(static_cast<uint32_t>(tc_compressed_code));
      triggerCell.setUncompressedCharge(tc_value);
      GlobalPoint point = geometry()->getTriggerCellPosition(tc_id);

      // 'value' is hardware, so p4 is meaningless, except for eta and phi
      math::PtEtaPhiMLorentzVector p4((double)tc_compressed_value / cosh(point.eta()), point.eta(), point.phi(), 0.);
      triggerCell.setP4(p4);
      triggerCell.setPosition(point);

      // calibration
      if (triggerCell.hwPt() > 0) {
        l1t::HGCalTriggerCell calibratedtriggercell(triggerCell);
        if (isNose) {
          calibrationNose_->calibrateInGeV(calibratedtriggercell);
        } else if (isSilicon) {
          if (isEM) {
            calibrationEE_->calibrateInGeV(calibratedtriggercell);
          } else {
            calibrationHEsi_->calibrateInGeV(calibratedtriggercell);
          }
        } else {
          calibrationHEsc_->calibrateInGeV(calibratedtriggercell);
        }
        triggerCellColl.push_back(0, calibratedtriggercell);
      }
    }
  }
}
