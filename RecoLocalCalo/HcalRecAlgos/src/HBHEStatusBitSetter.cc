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
					 std::vector<edm::ParameterSet> pulseShapeParameterSets)
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

void HBHEStatusBitSetter::SetFlagsFromDigi(HBHERecHit& hbhe, const HBHEDataFrame& digi)
{
  //clear status
  hbhe.setFlags(0);
  
  //increment hit multiplicity
  if (hbhe.energy()>hitEnergyMinimum_) {
    int index=logicalMap_->getHcalFrontEndId(hbhe.detid()).rmIndex();
    hpdMultiplicity_.at(index)++;
  }

  //set pulse shape bit
  double charge_total=0.0;
  double charge_max3=-100.0;
  double charge_late3=-100.0;
  unsigned int slice_max3=0;
  unsigned int size=digi.size();
  for (unsigned int iSlice=0;iSlice<size;iSlice++) {
    charge_total+=digi[iSlice].nominal_fC()-nominalPedestal_;
    if (iSlice<2) continue;
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
    if (charge_total<pulseShapeParameters_[iCut].at(0) || charge_total>=pulseShapeParameters_[iCut].at(1)) continue;
    if ( charge_late3< (pulseShapeParameters_[iCut].at(2)+charge_total*pulseShapeParameters_[iCut].at(3)) ) continue;
    if ( charge_late3>=(pulseShapeParameters_[iCut].at(4)+charge_total*pulseShapeParameters_[iCut].at(5)) ) continue;
    hbhe.setFlags(hbhe.flags()|(1<<1));
    return;
  }
  
}

void HBHEStatusBitSetter::SetFlagsFromRecHits(HBHERecHitCollection& rec) {
  for (HBHERecHitCollection::iterator iHBHE=rec.begin();iHBHE!=rec.end();++iHBHE) {
    int index=logicalMap_->getHcalFrontEndId(iHBHE->detid()).rmIndex();
    if (hpdMultiplicity_.at(index)<hitMultiplicityThreshold_) continue;
    iHBHE->setFlags(iHBHE->flags()|(1<<0));
  }
}
