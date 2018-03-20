#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSelectionImpl.h"

HGCalConcentratorSelectionImpl::
HGCalConcentratorSelectionImpl(const edm::ParameterSet& conf):
	nData_(conf.getParameter<uint32_t>("NData")),
	nCellsInModule_(conf.getParameter<uint32_t>("MaxCellsInModule")),
	linLSB_(conf.getParameter<double>("linLSB")),
	TCThreshold_fC_(conf.getParameter<double>("TCThreshold_fC"))
{  
  // Cannot have more selected cells than the max number of cells
  if(nData_>nCellsInModule_) nData_ = nCellsInModule_;
  TCThreshold_ADC_ = (int) (TCThreshold_fC_ / linLSB_);
}

void 
HGCalConcentratorSelectionImpl::
thresholdSelectImpl(std::vector<l1t::HGCalTriggerCell>& trigCellVec)
{
  for (size_t i = 0; i<trigCellVec.size();i++){
    int threshold = (HGCalDetId(trigCellVec[i].detId()).subdetId()==ForwardSubdetector::HGCHEB ? TCThresholdBH_ADC_ : TCThreshold_ADC_);
    if (trigCellVec[i].hwPt() < threshold)  trigCellVec[i].setHwPt(0);
  } 
}

void 
HGCalConcentratorSelectionImpl::
bestChoiceSelectImpl(std::vector<l1t::HGCalTriggerCell>& trigCellVec)
{
    // sort, reverse order
    sort(trigCellVec.begin(), trigCellVec.end(),
            [](const l1t::HGCalTriggerCell& a, 
                const  l1t::HGCalTriggerCell& b) -> bool
            { 
                return a.hwPt() > b.hwPt(); 
            } 
            );
    // keep only the first trigger cells
    if(trigCellVec.size()>nData_) trigCellVec.resize(nData_);
}



