#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSuperTriggerCellImpl.h"

#include <iostream>
#include <unordered_map>

HGCalConcentratorSuperTriggerCellImpl::
HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf):
  nData_(conf.getParameter<uint32_t>("NData")),
  nCellsInModule_(conf.getParameter<uint32_t>("MaxCellsInModule")),
  linLSB_(conf.getParameter<double>("linLSB")),
  adcsaturationBH_(conf.getParameter<double>("adcsaturationBH")),
  adcnBitsBH_(conf.getParameter<uint32_t>("adcnBitsBH")),
  TCThreshold_fC_(conf.getParameter<double>("TCThreshold_fC")),
  TCThresholdBH_MIP_(conf.getParameter<double>("TCThresholdBH_MIP")),
  triggercell_threshold_silicon_( conf.getParameter<double>("triggercell_threshold_silicon") ),
  triggercell_threshold_scintillator_( conf.getParameter<double>("triggercell_threshold_scintillator"))
{
  // Cannot have more selected cells than the max number of cells
  if(nData_>nCellsInModule_) nData_ = nCellsInModule_;
  adcLSBBH_ =  adcsaturationBH_/pow(2.,adcnBitsBH_);
  TCThreshold_ADC_ = (int) (TCThreshold_fC_ / linLSB_);
  TCThresholdBH_ADC_ = (int) (TCThresholdBH_MIP_ / adcLSBBH_);
}


int
HGCalConcentratorSuperTriggerCellImpl::getSuperTriggerCellId_(int detid) const {
  HGCalDetId TC_id(detid);
  if(TC_id.subdetId()==HGCHEB) {
    return TC_id.cell(); //scintillator
  } else {
    int TC_wafer = TC_id.wafer();
    int TC_12th = ( TC_id.cell() & 0x3a );
    
    return TC_wafer<<6 | TC_12th;
  }
  
}

void 
HGCalConcentratorSuperTriggerCellImpl::
superTriggerCellSelectImpl(uint32_t module_id ,const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput)
{ 

  std::unordered_map<int,SuperTriggerCell> STCs;

  // first pass, fill the super trigger cells
  for (const l1t::HGCalTriggerCell & tc : trigCellVecInput) {
    if (tc.subdetId() == HGCHEB) continue;
    STCs[getSuperTriggerCellId_(tc.detId())].add(tc);
  }
    
  // second pass, write them out
  for (const l1t::HGCalTriggerCell & tc : trigCellVecInput) {

    //If scintillator use a simple threshold cut
    if (tc.subdetId() == HGCHEB) {
      if ( ( tc.hwPt() >= TCThresholdBH_ADC_ ) && (tc.mipPt() >= triggercell_threshold_scintillator_ ) ) {
	trigCellVecOutput.push_back( tc );
      }
    } else {
      const auto & stc = STCs[getSuperTriggerCellId_(tc.detId())]; 
      if (tc.detId() == stc.maxId) {
	trigCellVecOutput.push_back( tc );
        stc.assignEnergy(trigCellVecOutput.back());
      }
    }

  } // end of second loop
  
}
