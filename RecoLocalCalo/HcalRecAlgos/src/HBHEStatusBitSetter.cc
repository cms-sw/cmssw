#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEStatusBitSetter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

HBHEStatusBitSetter::HBHEStatusBitSetter()
{
  logicalMap_=0;

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
					 const std::vector<edm::ParameterSet>& pulseShapeParameterSets
					 )
{
  logicalMap_=0;
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
  if (logicalMap_!=0) delete logicalMap_;
}

void HBHEStatusBitSetter::Clear()
{
  for (unsigned int i=0;i<hpdMultiplicity_.size();i++) hpdMultiplicity_[i]=0;
}

void HBHEStatusBitSetter::SetFlagsFromDigi(const HcalTopology* topo, HBHERecHit& hbhe, 
					   const HBHEDataFrame& digi,
					   const HcalCoder& coder,
					   const HcalCalibrations& calib,
					   int firstSample,
					   int samplesToAdd
					   )
{
  if (logicalMap_==0) {
    HcalLogicalMapGenerator gen;
    logicalMap_=new HcalLogicalMap(gen.createMap(topo));
  }
  

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
  //double shuichi_charge_total=0.0;
  double nominal_charge_total=0.0;  
  double charge_max3=-100.0;
  double charge_late3=-100.0;
  unsigned int slice_max3=0;
  unsigned int size=digi.size();
 
  CaloSamples tool;
  coder.adc2fC(digi,tool);

  //  int capid=-1;
  for (unsigned int iSlice=0;iSlice<size;iSlice++) 
    {
      //      capid  = digi.sample(iSlice).capid();
      //shuichi_charge_total+=tool[iSlice]-calib.pedestal(capid);
      nominal_charge_total+=digi[iSlice].nominal_fC()-nominalPedestal_;

      if (iSlice<2) continue;
      // digi[i].nominal_fC() could be replaced by tool[iSlice], I think...  -- Jeff, 11 April 2011
      double qsum3=digi[iSlice].nominal_fC() + digi[iSlice-1].nominal_fC() + digi[iSlice-2].nominal_fC() - 3*nominalPedestal_;
      if (qsum3>charge_max3) {
	charge_max3=qsum3;
	slice_max3=iSlice;
      }
    }

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

void HBHEStatusBitSetter::SetFlagsFromRecHits(const HcalTopology* topo, HBHERecHitCollection& rec) {

  if (logicalMap_==0) {
    HcalLogicalMapGenerator gen;
    logicalMap_=new HcalLogicalMap(gen.createMap(topo));
  }


  for (HBHERecHitCollection::iterator iHBHE=rec.begin();iHBHE!=rec.end();++iHBHE) {
    int index=logicalMap_->getHcalFrontEndId(iHBHE->detid()).rmIndex();
    if (hpdMultiplicity_.at(index)<hitMultiplicityThreshold_) continue;
    iHBHE->setFlagField(1,HcalCaloFlagLabels::HBHEHpdHitMultiplicity);
  }
}
