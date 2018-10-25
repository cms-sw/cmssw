#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSelectionImpl.h"

HGCalConcentratorSelectionImpl::
HGCalConcentratorSelectionImpl(const edm::ParameterSet& conf):
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

void 
HGCalConcentratorSelectionImpl::
thresholdSelectImpl(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput)
{ 
  for (size_t i = 0; i<trigCellVecInput.size();i++){

    int threshold = (HGCalDetId(trigCellVecInput[i].detId()).subdetId()==ForwardSubdetector::HGCHEB ? TCThresholdBH_ADC_ : TCThreshold_ADC_);
    double triggercell_threshold = (HGCalDetId(trigCellVecInput[i].detId()).subdetId()==HGCHEB ? triggercell_threshold_scintillator_ : triggercell_threshold_silicon_);
  
    if ((trigCellVecInput[i].hwPt() >= threshold) && (trigCellVecInput[i].mipPt() >= triggercell_threshold)){ 
      trigCellVecOutput.push_back(trigCellVecInput[i]);      
    }  
  }
}

void 
HGCalConcentratorSelectionImpl::
bestChoiceSelectImpl(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput)
{ 
  trigCellVecOutput = trigCellVecInput;    
  // sort, reverse order
  sort(trigCellVecOutput.begin(), trigCellVecOutput.end(),
       [](const l1t::HGCalTriggerCell& a, 
          const  l1t::HGCalTriggerCell& b) -> bool
  { 
    return a.hwPt() > b.hwPt(); 
  } 
  );

  // keep only the first trigger cells
  if(trigCellVecOutput.size()>nData_) trigCellVecOutput.resize(nData_);
  
}

l1t::HGCalSuperTriggerCellMap*
HGCalConcentratorSelectionImpl::
getSuperTriggerCell_(uint32_t module_id, l1t::HGCalTriggerCell TC)
{

  HGCalDetId TC_id( TC.detId() );
  int TC_wafer = TC_id.wafer();

  int TC_12th = ( TC_id.cell() & 0x3a );

  
  long SuperTriggerCellMap_id = 0;
  if(TC_id.subdetId()==HGCHEB) SuperTriggerCellMap_id = TC_id.cell(); //scintillator

  else SuperTriggerCellMap_id = TC_wafer<<6 | TC_12th;

  return &mapSuperTriggerCellMap_[module_id][SuperTriggerCellMap_id];

}

void 
HGCalConcentratorSelectionImpl::
clearSuperTriggerCellMap(){
  mapSuperTriggerCellMap_.clear();
}


void 
HGCalConcentratorSelectionImpl::
superTriggerCellSelectImpl(uint32_t module_id ,const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput)
{ 

  //Clear SuperTriggerCell Map
  this->clearSuperTriggerCellMap();

  for (size_t i = 0; i<trigCellVecInput.size();i++){

    getSuperTriggerCell_( module_id, trigCellVecInput.at(i) )->addTriggerCell( trigCellVecInput.at(i) );

  }
    
  for (size_t i = 0; i<trigCellVecInput.size();i++){

    //If scintillator use a simple threshold cut
    if ( HGCalDetId(trigCellVecInput[i].detId()).subdetId()==HGCHEB ){

      if  ( ( trigCellVecInput[i].hwPt() >= TCThresholdBH_ADC_ ) && (trigCellVecInput[i].mipPt() >= triggercell_threshold_scintillator_ ) ){

	trigCellVecOutput.push_back( trigCellVecInput.at(i) );

      }

    }

    else{
   
      l1t::HGCalSuperTriggerCellMap* TCmap = getSuperTriggerCell_( module_id, trigCellVecInput.at(i) );    

      
      //Check if TC is the most energetic in superTC and assign the full hwPt of the superTC
      //Else zeroed
      
      if(TCmap->maxTriggerCell().detId() == trigCellVecInput.at(i).detId()) {
		
	trigCellVecOutput.push_back( trigCellVecInput.at(i) );
	trigCellVecOutput.back().setHwPt(TCmap->hwPt());
	
      }

    }
    
  }  
  
}
