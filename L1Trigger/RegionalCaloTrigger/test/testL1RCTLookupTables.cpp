#include <vector>
#include <iostream>
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
using std::cout;
using std::endl;
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
  std::cout << lut->lookup(0,0,0,0,0,0) << " should equal 0" << std::endl;
  std::cout << lut->lookup(2,0,0,0,0,0) << " should equal 514" << std::endl;
  std::cout << lut->lookup(10,0,0,0,0,0) << " should equal 133642 " << std::endl;
  std::cout << lut->lookup(10,0,1,0,0,0) << " should equal 133770 " << std::endl;
  std::cout << lut->lookup(0,10,0,0,0,0) << " should equal 133770 " << std::endl;
  std::cout << lut->lookup(0,10,1,0,0,0) << " should equal 133770 " << std::endl;
  std::cout << lut->lookup(255,0,0,0,0,0) << " should equal 196479 " << std::endl;
}
  
