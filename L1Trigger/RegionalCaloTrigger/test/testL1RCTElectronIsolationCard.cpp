#include <vector>
#include <iostream>

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTElectronIsolationCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

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
  L1RCTElectronIsolationCard eic(0,0,lut);
  L1RCTRegion r0;
  L1RCTRegion r1;
  //This should report 1 isolated electron of 100 
  r0.setEtIn7Bits(0,0,100);
  eic.setRegion(0,r0);
  eic.setRegion(1,r1);

  eic.fillElectronCandidates();
  eic.print();

  //This should report *no* electrons
  r0.setHE_FGBit(0,0,1);
  eic.fillElectronCandidates();
  eic.print();

  //This should report only a nonisolated electron of 100
  r0.setHE_FGBit(0,0,0);
  r0.setHE_FGBit(0,1,1);
  eic.fillElectronCandidates();
  eic.print();

  //This should report an isolated electron of 80 and a nonisolated of 100
  r0.setEtIn7Bits(2,0,80);
  eic.fillElectronCandidates();
  eic.print();
			     

}
