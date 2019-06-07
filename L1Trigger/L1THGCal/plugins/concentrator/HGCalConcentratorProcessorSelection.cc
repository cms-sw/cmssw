#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorProcessorSelection.h"
#include <limits>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

DEFINE_EDM_PLUGIN(HGCalConcentratorFactory, HGCalConcentratorProcessorSelection, "HGCalConcentratorProcessorSelection");

HGCalConcentratorProcessorSelection::HGCalConcentratorProcessorSelection(const edm::ParameterSet& conf)
    : HGCalConcentratorProcessorBase(conf),
      fixedDataSizePerHGCROC_(conf.getParameter<bool>("fixedDataSizePerHGCROC")),
      coarsenTriggerCells_(conf.getParameter<bool>("coarsenTriggerCells")) {
  std::string selectionType(conf.getParameter<std::string>("Method"));
  if (selectionType == "thresholdSelect") {
    selectionType_ = thresholdSelect;
    thresholdImpl_ = std::make_unique<HGCalConcentratorThresholdImpl>(conf);
  } else if (selectionType == "bestChoiceSelect") {
    selectionType_ = bestChoiceSelect;
    bestChoiceImpl_ = std::make_unique<HGCalConcentratorBestChoiceImpl>(conf);
  } else if (selectionType == "superTriggerCellSelect") {
    selectionType_ = superTriggerCellSelect;
    superTriggerCellImpl_ = std::make_unique<HGCalConcentratorSuperTriggerCellImpl>(conf);
  } else {
    throw cms::Exception("HGCTriggerParameterError")
        << "Unknown type of concentrator selection '" << selectionType << "'";
  }

  if (coarsenTriggerCells_ || fixedDataSizePerHGCROC_) {
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

    int thickness = 0;
    if (triggerTools_.isSilicon(module_trigcell.second.at(0).detId())) {
      thickness = triggerTools_.thicknessIndex(module_trigcell.second.at(0).detId(), true);
    } else if (triggerTools_.isScintillator(module_trigcell.second.at(0).detId())) {
      thickness = 3;
    }

    if (coarsenTriggerCells_ || (fixedDataSizePerHGCROC_ && thickness > kHighDensityThickness_)) {
      coarsenerImpl_->coarsen(module_trigcell.second, trigCellVecCoarsened);

      switch (selectionType_) {
        case thresholdSelect:
          thresholdImpl_->select(trigCellVecCoarsened, trigCellVecOutput);
          break;
        case bestChoiceSelect:
          bestChoiceImpl_->select(geometry_->getLinksInModule(module_trigcell.first),
                                  geometry_->getModuleSize(module_trigcell.first),
                                  trigCellVecCoarsened,
                                  trigCellVecOutput);
          break;
        case superTriggerCellSelect:
          superTriggerCellImpl_->select(trigCellVecCoarsened, trigCellVecOutput);
          break;
        default:
          // Should not happen, selection type checked in constructor
          break;
      }

    } else {
      switch (selectionType_) {
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
