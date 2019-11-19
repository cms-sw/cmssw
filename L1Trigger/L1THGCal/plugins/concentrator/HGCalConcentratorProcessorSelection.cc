#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorProcessorSelection.h"
#include <limits>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

DEFINE_EDM_PLUGIN(HGCalConcentratorFactory, HGCalConcentratorProcessorSelection, "HGCalConcentratorProcessorSelection");

HGCalConcentratorProcessorSelection::HGCalConcentratorProcessorSelection(const edm::ParameterSet& conf)
    : HGCalConcentratorProcessorBase(conf),
      fixedDataSizePerHGCROC_(conf.getParameter<bool>("fixedDataSizePerHGCROC")),
      coarsenTriggerCells_(conf.getParameter<std::vector<unsigned>>("coarsenTriggerCells")),
      selectionType_(kNSubDetectors_) {
  std::vector<std::string> selectionType(conf.getParameter<std::vector<std::string>>("Method"));
  if (selectionType.size() != kNSubDetectors_ || coarsenTriggerCells_.size() != kNSubDetectors_) {
    throw cms::Exception("HGCTriggerParameterError")
        << "Inconsistent number of sub-detectors (should be " << kNSubDetectors_ << ")";
  }

  for (int subdet = 0; subdet < kNSubDetectors_; subdet++) {
    if (selectionType[subdet] == "thresholdSelect") {
      selectionType_[subdet] = thresholdSelect;
      if (!thresholdImpl_)
        thresholdImpl_ = std::make_unique<HGCalConcentratorThresholdImpl>(conf);
    } else if (selectionType[subdet] == "bestChoiceSelect") {
      selectionType_[subdet] = bestChoiceSelect;
      if (!bestChoiceImpl_)
        bestChoiceImpl_ = std::make_unique<HGCalConcentratorBestChoiceImpl>(conf);
    } else if (selectionType[subdet] == "superTriggerCellSelect") {
      selectionType_[subdet] = superTriggerCellSelect;
      if (!superTriggerCellImpl_)
        superTriggerCellImpl_ = std::make_unique<HGCalConcentratorSuperTriggerCellImpl>(conf);
    } else if (selectionType[subdet] == "noSelection") {
      selectionType_[subdet] = noSelection;
    } else {
      throw cms::Exception("HGCTriggerParameterError")
          << "Unknown type of concentrator selection '" << selectionType[subdet] << "'";
    }
  }

  if (std::find(coarsenTriggerCells_.begin(), coarsenTriggerCells_.end(), true) != coarsenTriggerCells_.end() ||
      fixedDataSizePerHGCROC_) {
    coarsenerImpl_ = std::make_unique<HGCalConcentratorCoarsenerImpl>(conf);
  }
}

void HGCalConcentratorProcessorSelection::run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& triggerCellCollInput,
                                              l1t::HGCalTriggerCellBxCollection& triggerCellCollOutput,
                                              const edm::EventSetup& es) {
  if (thresholdImpl_)
    thresholdImpl_->eventSetup(es);
  if (bestChoiceImpl_)
    bestChoiceImpl_->eventSetup(es);
  if (superTriggerCellImpl_)
    superTriggerCellImpl_->eventSetup(es);
  if (coarsenerImpl_)
    coarsenerImpl_->eventSetup(es);
  triggerTools_.eventSetup(es);

  const l1t::HGCalTriggerCellBxCollection& collInput = *triggerCellCollInput;

  std::unordered_map<uint32_t, std::vector<l1t::HGCalTriggerCell>> tc_modules;
  for (const auto& trigCell : collInput) {
    uint32_t module = geometry_->getModuleFromTriggerCell(trigCell.detId());
    tc_modules[module].push_back(trigCell);
  }

  for (const auto& module_trigcell : tc_modules) {
    std::vector<l1t::HGCalTriggerCell> trigCellVecOutput;
    std::vector<l1t::HGCalTriggerCell> trigCellVecCoarsened;

    int thickness = triggerTools_.thicknessIndex(module_trigcell.second.at(0).detId(), true);

    HGCalTriggerTools::SubDetectorType subdet = triggerTools_.getSubDetectorType(module_trigcell.second.at(0).detId());

    if (coarsenTriggerCells_[subdet] || (fixedDataSizePerHGCROC_ && thickness > kHighDensityThickness_)) {
      coarsenerImpl_->coarsen(module_trigcell.second, trigCellVecCoarsened);

      switch (selectionType_[subdet]) {
        case thresholdSelect:
          thresholdImpl_->select(trigCellVecCoarsened, trigCellVecOutput);
          break;
        case bestChoiceSelect:
          if (triggerTools_.isEm(module_trigcell.first)) {
            bestChoiceImpl_->select(geometry_->getLinksInModule(module_trigcell.first),
                                    geometry_->getModuleSize(module_trigcell.first),
                                    module_trigcell.second,
                                    trigCellVecOutput);
          } else {
            bestChoiceImpl_->select(geometry_->getLinksInModule(module_trigcell.first),
                                    geometry_->getModuleSize(module_trigcell.first),
                                    trigCellVecCoarsened,
                                    trigCellVecOutput);
          }
          break;
        case superTriggerCellSelect:
          superTriggerCellImpl_->select(trigCellVecCoarsened, trigCellVecOutput);
          break;
        case noSelection:
          trigCellVecOutput = trigCellVecCoarsened;
          break;
        default:
          // Should not happen, selection type checked in constructor
          break;
      }

    } else {
      switch (selectionType_[subdet]) {
        case thresholdSelect:
          thresholdImpl_->select(module_trigcell.second, trigCellVecOutput);
          break;
        case bestChoiceSelect:
          bestChoiceImpl_->select(geometry_->getLinksInModule(module_trigcell.first),
                                  geometry_->getModuleSize(module_trigcell.first),
                                  module_trigcell.second,
                                  trigCellVecOutput);
          break;
        case superTriggerCellSelect:
          superTriggerCellImpl_->select(module_trigcell.second, trigCellVecOutput);
          break;
        case noSelection:
          trigCellVecOutput = module_trigcell.second;
          break;
        default:
          // Should not happen, selection type checked in constructor
          break;
      }
    }

    for (const auto& trigCell : trigCellVecOutput) {
      triggerCellCollOutput.push_back(0, trigCell);
    }
  }
}
