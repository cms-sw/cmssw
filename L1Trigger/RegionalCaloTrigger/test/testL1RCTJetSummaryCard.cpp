#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTJetSummaryCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

int main() {
  std::string filename("../data/TPGcalc.txt");
  L1RCTLookupTables lut(filename);
  L1RCTJetSummaryCard jsc(0);
  vector<unsigned short> hfregions(8);
  vector<unsigned short> bregions(14);
  vector<unsigned short> tauBits(14);
  vector<unsigned short> mipBits(14);
  vector<unsigned short> isoElectrons(14);
  vector<unsigned short> nonIsoElectrons(14);
  isoElectrons.at(0) = 10;
  isoElectrons.at(1) = 20;
  isoElectrons.at(2) = 30;
  isoElectrons.at(3) = 40;
  isoElectrons.at(4) = 50;
  nonIsoElectrons.at(0) = 80;
  nonIsoElectrons.at(1) = 35;
  nonIsoElectrons.at(2) = 92;
  nonIsoElectrons.at(3) = 50;
  nonIsoElectrons.at(4) = 49;
  nonIsoElectrons.at(5) = 34;
  mipBits.at(0) = 1;
  mipBits.at(1) = 1;
  mipBits.at(10) = 1;
  bregions.at(0) = 100;
  bregions.at(2) = 50;
  bregions.at(12) = 50;
  jsc.fillMIPBits(mipBits);
  jsc.fillTauBits(tauBits);
  jsc.fillNonIsolatedEGObjects(nonIsoElectrons);
  jsc.fillIsolatedEGObjects(isoElectrons);
  jsc.fillRegionSums(bregions);
  jsc.fillHFRegionSums(hfregions,&lut);
  jsc.fillQuietBits();
  jsc.fillJetRegions();
  jsc.print();
}
