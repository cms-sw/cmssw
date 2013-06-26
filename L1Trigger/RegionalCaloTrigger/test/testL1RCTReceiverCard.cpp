#include <vector>

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTReceiverCard.h"
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
  L1RCTReceiverCard flip(9,0,lut);
  L1RCTReceiverCard card(0,0,lut);
  L1RCTReceiverCard six(0,6,lut);
  std::vector<unsigned short> input1(64);
  std::vector<unsigned short> input2(64);
  std::vector<unsigned short> input3(64);
  std::vector<unsigned short> input4(64);
  input1.at(0) = 100;
  input1.at(1) = 100;
  input1.at(7) = 100;
  //All these inputs should go into the ecal.
  //They should be at positions 
  //0  0  0    100
  //0  0  0    100
  //0  0  0    0
  //0  0  100  0
  //The energy sum should be 300 and
  //the tau bit should be set to on because
  //the phi pattern is 1101
  card.fillInput(input1);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  card.print();
  

  //The following should look like
  //0 0 0 100
  //0 0 0 0
  //0 0 0 0
  //0 0 0 0
  //The tau bit should be off.
  input2.at(0) = 50;
  input2.at(32) = 50;
  card.fillInput(input2);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  card.print();
  
  //The following should look like
  //0 0 0 100
  //0 0 0 100
  //0 0 0 0
  //0 0 0 0
  //and have the muon bit on at tower 0
  //and have the fg bit on at tower 1 and the he bit on at tower 0
  //because the h is greater than the e.  The muon bit for region 0
  //should be on and the tau bit should be off.
  input3.at(32) = 356;
  input3.at(1) = 356;
  card.fillInput(input3);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  card.print();
  
  //Let's make sure that everything can be set correctly in all the regions
  //We'll set the energies to be the same as the tower number+1 and
  //make sure it matches with the layout of the receiver card that
  //we want, kthnx.

  for(int i =0;i<32;i++)
    input4.at(i)=i+1;
  card.fillInput(input4);
  card.fillMuonBits();
  card.fillTauBits();
  card.fillRegionSums();
  card.print();

  flip.fillInput(input4);
  flip.fillMuonBits();
  flip.fillTauBits();
  flip.fillRegionSums();
  flip.print();

  six.fillInput(input4);
  six.fillMuonBits();
  six.fillTauBits();
  six.fillRegionSums();
  six.print();

}
