#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTORCAMap.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

using std::vector;
using std::fstream;
using std::cout;
using std::endl;
using std::ios;

int main (){
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
  L1RCT rct(lut);
  std::vector<int> data(4);
  std::vector<int> location(3);
  unsigned long lookupValue;
  std::vector<std::vector<unsigned short> > hf(18,std::vector<unsigned short>(8));
  std::vector<std::vector<std::vector<unsigned short> > > barrel(18,std::vector<std::vector<unsigned short> >(7,
                                            std::vector<unsigned short>(64)));

  char throwaway[2000];
  //Now we pull in the data from the crate.input file
  fstream input("crate.input",std::ios::in);
  if ( ! input.is_open() ) return 0;

  fstream output("lut.out",std::ios::out);
  fstream rctoutput("rct.out",std::ios::out);
  input.getline(throwaway,2000);
  L1RCTORCAMap theMap;
  output << "Eta    Phi    Ecal    Hcal    Lookup" << std::endl;
  while(!input.eof()){
    for(int i=0;i<4;i++){
      input >> data.at(i);
      output << data.at(i) << "      ";
    }
      location = theMap.orcamap(data.at(0),data.at(1));
      barrel.at(location.at(0)).at(location.at(1)).at(location.at(2)) = data.at(2);
      barrel.at(location.at(0)).at(location.at(1)).at(location.at(2)+32) = data.at(3);
      lookupValue = lut->lookup(data.at(2)&255,data.at(3)&255,(data.at(2)<<8)&1,
				location.at(0), location.at(1), location.at(2));
      output << lookupValue << std::endl;
  }
  rct.input(barrel,hf);
  rct.processEvent();
  input.close();
  output.close();
  rct.printCrate(0);
}
