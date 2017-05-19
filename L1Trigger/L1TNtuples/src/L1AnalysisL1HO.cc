#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1HO.h"
L1Analysis::L1AnalysisL1HO::L1AnalysisL1HO()
{
}

L1Analysis::L1AnalysisL1HO::~L1AnalysisL1HO()
{
}

void L1Analysis::L1AnalysisL1HO::SetHO(const edm::SortedCollection<HODataFrame>& hoDataFrame)
{
  for (edm::SortedCollection<HODataFrame>::const_iterator it = hoDataFrame.begin(); it != hoDataFrame.end(); ++it){
    HcalDetId hcalDetId = it->id();

    for (int i = 0; i < it->size(); ++i) {
      HcalQIESample hcalQIESample = it->sample(i);
      l1ho_.hcalDetIdIEta.push_back(hcalDetId.ieta());
      l1ho_.hcalDetIdIPhi.push_back(hcalDetId.iphi());
      l1ho_.hcalQIESample.push_back(i);
      l1ho_.hcalQIESampleAdc.push_back(hcalQIESample.adc());
      l1ho_.hcalQIESampleDv.push_back(hcalQIESample.dv());
      l1ho_.hcalQIESampleEr.push_back(hcalQIESample.er());

      ++l1ho_.nHcalQIESamples;
    }

    ++l1ho_.nHcalDetIds;
  }
}

