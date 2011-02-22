#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEStatusBitSetter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

HBHEStatusBitSetter::HBHEStatusBitSetter()
{
  HcalLogicalMapGenerator gen;
  logicalMap_=new HcalLogicalMap(gen.createMap());

  for (int iRm=0;iRm<HcalFrontEndId::maxRmIndex;iRm++) {
    hpdMultiplicity_.push_back(0);
  }

  nominalPedestal_=3.0;
  hitEnergyMinimum_=2.0;
  hitMultiplicityThreshold_=17;
}

HBHEStatusBitSetter::HBHEStatusBitSetter(double nominalPedestal,
					 double hitEnergyMinimum,
					 int hitMultiplicityThreshold,
					 std::vector<edm::ParameterSet> pulseShapeParameterSets
					 )
{
  HcalLogicalMapGenerator gen;
  logicalMap_=new HcalLogicalMap(gen.createMap());
  
  for (int iRm=0;iRm<HcalFrontEndId::maxRmIndex;iRm++) {
    hpdMultiplicity_.push_back(0);
  }

  nominalPedestal_=nominalPedestal;
  hitEnergyMinimum_=hitEnergyMinimum;
  hitMultiplicityThreshold_=hitMultiplicityThreshold;

  for (unsigned int iPSet=0;iPSet<pulseShapeParameterSets.size();iPSet++) {
    edm::ParameterSet pset=pulseShapeParameterSets.at(iPSet);
    std::vector<double> params=pset.getParameter<std::vector<double> >("pulseShapeParameters");
    pulseShapeParameters_.push_back(params);
  }

}

HBHEStatusBitSetter::~HBHEStatusBitSetter() {
  delete logicalMap_;
}

void HBHEStatusBitSetter::Clear()
{
  for (unsigned int i=0;i<hpdMultiplicity_.size();i++) hpdMultiplicity_[i]=0;
}

void HBHEStatusBitSetter::SetFlagsFromDigi(HBHERecHit& hbhe, 
					   const HBHEDataFrame& digi,
					   const HcalCoder& coder,
					   const HcalCalibrations& calib,
					   int firstSample,
					   int samplesToAdd
					   )
{
  // get firstSample, samplesToAdd from database for each hit
  firstSample_ = firstSample;
  samplesToAdd_ = samplesToAdd;

  //increment hit multiplicity
  if (hbhe.energy()>hitEnergyMinimum_) {
    int index=logicalMap_->getHcalFrontEndId(hbhe.detid()).rmIndex();
    hpdMultiplicity_.at(index)++;
  }

  //set pulse shape bits
  // Shuichi's algorithm uses the "correct" charge & pedestals, while Ted's uses "nominal" values.
  // Perhaps we should correct Ted's algorithm in the future, though that will mean re-tuning thresholds for his cuts. -- Jeff, 28 May 2010
  double charge_total=0.0;
  double nominal_charge_total=0.0;  
  double charge_max3=-100.0;
  double charge_late3=-100.0;
  unsigned int slice_max3=0;
  unsigned int size=digi.size();
 
  CaloSamples tool;
  coder.adc2fC(digi,tool);

  double max2TS=-99.; // largest 2-TS sum found in reco window so far
  int max2TS_counter=1; // default value is 1
  double running2TS=0; // tracks current 2-TS sum, using "correct" charge and pedestal
  int capid=-1, capid2=-1;
  for (unsigned int iSlice=0;iSlice<size;iSlice++) 
    {
      capid  = digi.sample(iSlice).capid();
      charge_total+=tool[iSlice]-calib.pedestal(capid);
      nominal_charge_total+=digi[iSlice].nominal_fC()-nominalPedestal_;
      if (iSlice>=firstSample_ && 
	  iSlice<(firstSample_ + samplesToAdd_-1) && 
	  iSlice<(size-1))
	{
	  capid2 = digi.sample(iSlice+1).capid();
	  running2TS=tool[iSlice]+tool[iSlice+1]-calib.pedestal(capid)-calib.pedestal(capid2);
	  if (running2TS>max2TS)
	    max2TS=running2TS;
	}
      if (iSlice<2) continue;
      double qsum3=digi[iSlice].nominal_fC() + digi[iSlice-1].nominal_fC() + digi[iSlice-2].nominal_fC() - 3*nominalPedestal_;
      if (qsum3>charge_max3) {
	charge_max3=qsum3;
	slice_max3=iSlice;
      }
    }

  // max2TS counter will be set from 1->63, indicating the fraction of total charge
  // contained within the largest 2 time slices within the reco window.

  if (charge_total<0 && max2TS>0)
    max2TS_counter=63;
      
  else if (charge_total>0)
    {
      max2TS_counter=int(100*(max2TS/charge_total-0.5))+1;
      if (max2TS_counter<1) max2TS_counter=1;
      if (max2TS_counter>63) max2TS_counter=63;
    }
      
  hbhe.setFlagField(max2TS_counter, HcalCaloFlagLabels::Fraction2TS,6);
  
  if ((4+slice_max3)>size) return;
  charge_late3=digi[slice_max3+1].nominal_fC() + digi[slice_max3+2].nominal_fC() + digi[slice_max3+3].nominal_fC() - 3*nominalPedestal_;

  for (unsigned int iCut=0;iCut<pulseShapeParameters_.size();iCut++) {
    if (pulseShapeParameters_[iCut].size()!=6) continue;
    if (nominal_charge_total<pulseShapeParameters_[iCut].at(0) || nominal_charge_total>=pulseShapeParameters_[iCut].at(1)) continue;
    if ( charge_late3< (pulseShapeParameters_[iCut].at(2)+nominal_charge_total*pulseShapeParameters_[iCut].at(3)) ) continue;
    if ( charge_late3>=(pulseShapeParameters_[iCut].at(4)+nominal_charge_total*pulseShapeParameters_[iCut].at(5)) ) continue;
    hbhe.setFlagField(1,HcalCaloFlagLabels::HBHEPulseShape);
    return;
  }
  
}

void HBHEStatusBitSetter::SetFlagsFromRecHits(HBHERecHitCollection& rec) {
  for (HBHERecHitCollection::iterator iHBHE=rec.begin();iHBHE!=rec.end();++iHBHE) {
    int index=logicalMap_->getHcalFrontEndId(iHBHE->detid()).rmIndex();
    if (hpdMultiplicity_.at(index)<hitMultiplicityThreshold_) continue;
    iHBHE->setFlagField(1,HcalCaloFlagLabels::HBHEHpdHitMultiplicity);
  }
}
