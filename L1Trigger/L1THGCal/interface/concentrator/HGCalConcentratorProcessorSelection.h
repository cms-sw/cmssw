#ifndef __L1Trigger_L1THGCal_HGCalConcentratorProcessorSelection_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorProcessorSelection_h__

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorThresholdImpl.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorBestChoiceImpl.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSuperTriggerCellImpl.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorCoarsenerImpl.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

class HGCalConcentratorProcessorSelection : public HGCalConcentratorProcessorBase {
private:
  enum SelectionType { thresholdSelect, bestChoiceSelect, superTriggerCellSelect, mixedBestChoiceSuperTriggerCell };

public:
  HGCalConcentratorProcessorSelection(const edm::ParameterSet& conf);

  void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& triggerCellCollInput,
           l1t::HGCalTriggerCellBxCollection& triggerCellCollOutput,
           const edm::EventSetup& es) override;

private:
  SelectionType selectionType_;
  bool fixedDataSizePerHGCROC_;
  bool coarsenTriggerCells_;
  static constexpr int kHighDensityThickness_ = 0;

  std::unique_ptr<HGCalConcentratorThresholdImpl> thresholdImpl_;
  std::unique_ptr<HGCalConcentratorBestChoiceImpl> bestChoiceImpl_;
  std::unique_ptr<HGCalConcentratorSuperTriggerCellImpl> superTriggerCellImpl_;
  std::unique_ptr<HGCalConcentratorCoarsenerImpl> coarsenerImpl_;

  HGCalTriggerTools triggerTools_;
};

#endif
