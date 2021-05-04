#ifndef __L1Trigger_L1THGCal_HGCalConcentratorProcessorSelection_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorProcessorSelection_h__

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorThresholdImpl.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorBestChoiceImpl.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSuperTriggerCellImpl.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorCoarsenerImpl.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorTrigSumImpl.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

#include <utility>

class HGCalConcentratorProcessorSelection : public HGCalConcentratorProcessorBase {
private:
  enum SelectionType { thresholdSelect, bestChoiceSelect, superTriggerCellSelect, noSelection };

public:
  HGCalConcentratorProcessorSelection(const edm::ParameterSet& conf);

  void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& triggerCellCollInput,
           std::pair<l1t::HGCalTriggerCellBxCollection, l1t::HGCalTriggerSumsBxCollection>& triggerCollOutput,
           const edm::EventSetup& es) override;

private:
  bool fixedDataSizePerHGCROC_;
  std::vector<unsigned> coarsenTriggerCells_;
  static constexpr int kHighDensityThickness_ = 0;
  static constexpr int kNSubDetectors_ = 3;

  std::vector<SelectionType> selectionType_;

  std::unique_ptr<HGCalConcentratorThresholdImpl> thresholdImpl_;
  std::unique_ptr<HGCalConcentratorBestChoiceImpl> bestChoiceImpl_;
  std::unique_ptr<HGCalConcentratorSuperTriggerCellImpl> superTriggerCellImpl_;
  std::unique_ptr<HGCalConcentratorCoarsenerImpl> coarsenerImpl_;
  std::unique_ptr<HGCalConcentratorTrigSumImpl> trigSumImpl_;

  HGCalTriggerTools triggerTools_;
};

#endif
