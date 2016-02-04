#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTJetSummaryCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include <vector>
using std::vector;

int main() {
  // For testing use 1:1 LUT
  std::vector<double> eGammaECalScaleFactors(32, 1.0);
  std::vector<double> eGammaHCalScaleFactors(32, 1.0);
  std::vector<double> jetMETECalScaleFactors(32, 1.0);
  std::vector<double> jetMETHCalScaleFactors(32, 1.0);
  std::vector<double> c,d,e,f,g,h;
  L1RCTParameters* rctParameters = 
    new L1RCTParameters(1.0,                       // eGammaLSB
			1.0,                       // jetMETLSB
			3.0,                       // eMinForFGCut
			40.0,                      // eMaxForFGCut
			0.5,                       // hOeCut
			1.0,                       // eMinForHoECut
			50.0,                      // eMaxForHoECut
			1.0,                       // hMinForHoECut
			2.0,                       // eActivityCut
			3.0,                       // hActivityCut
			3,                         // eicIsolationThreshold
                        3,                         // jscQuietThresholdBarrel
                        3,                         // jscQuietThresholdEndcap
			false,                     // noiseVetoHB
			false,                     // noiseVetoHEplus
			false,                     // noiseVetoHEminus
			false,                     // use Lindsey
			eGammaECalScaleFactors,
			eGammaHCalScaleFactors,
			jetMETECalScaleFactors,
			jetMETHCalScaleFactors,
			c,
			d,
			e,
			f,
			g,
			h
			);
  L1RCTLookupTables* lut = new L1RCTLookupTables();
  lut->setRCTParameters(rctParameters);  // transcoder and etScale are not used
  L1RCTJetSummaryCard jsc(0,lut);
  std::vector<unsigned short> hfregions(8);
  std::vector<unsigned short> bregions(14);
  std::vector<unsigned short> tauBits(14);
  std::vector<unsigned short> mipBits(14);
  std::vector<unsigned short> isoElectrons(14);
  std::vector<unsigned short> nonIsoElectrons(14);
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
  jsc.fillHFRegionSums(hfregions);
  jsc.fillQuietBits();
  jsc.fillJetRegions();
  jsc.print();
}
