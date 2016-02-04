#include <iostream>
#include <string>

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

int main()
{
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
  const unsigned short iPhiMax = 72;
  const short iAbsEtaMax = 32;
  for(unsigned short iPhi = 0; iPhi < iPhiMax; iPhi++)
    {
      for(short iEta = -iAbsEtaMax; iEta <= iAbsEtaMax; iEta++)
	{
	  unsigned short iCrate;
	  unsigned short iCard;
	  unsigned short iTower;
	  if(abs(iEta) > 28 && iPhi > 17) 
	    {
	      iCrate = 777;
	      iCard = 777;
	      iTower = 777;
	    }
	  else
	    if(iEta != 0)
	      {
		iCrate = rctParameters->calcCrate(iPhi, iEta);
		iCard = rctParameters->calcCard(iPhi, (unsigned short) abs(iEta));
		iTower = rctParameters->calcTower(iPhi, (unsigned short) abs(iEta));
		short jEta = rctParameters->calcIEta(iCrate, iCard, iTower);
		unsigned short jPhi = rctParameters->calcIPhi(iCrate, iCard, iTower);
		if(iEta != jEta || iPhi != jPhi)
		  std::cout << "Problem 1: "
			    << "\tiEta   = " << iEta << "\tjEta   = " << jEta 
			    << "\tiPhi   = " << iPhi << "\tjPhi   = " << jPhi 
			    << "\tcrate[][] = " << iCrate 
			    << "\tcard[][]  = " << iCard 
			    << "\ttower[][] = " << iTower
			    << std::endl;
	      }
	    else
	      {
		iCrate = 888;
		iCard = 888;
		iTower = 888;
	      }
	}
    }
  for(unsigned short iCrate = 0; iCrate < 18; iCrate++)
    {
      for(unsigned short iCard = 0; iCard < 7; iCard++)
	{
	  for(unsigned short iTower = 0; iTower < 32; iTower++)
	    {
	      short iEta = rctParameters->calcIEta(iCrate, iCard, iTower);
	      unsigned short iPhi = rctParameters->calcIPhi(iCrate, iCard, iTower);
	      short jCrate = rctParameters->calcCrate(iPhi, iEta);
	      short jCard = rctParameters->calcCard(iPhi, (unsigned short) abs(iEta));
	      short jTower = rctParameters->calcTower(iPhi, (unsigned short) abs(iEta));
	      if(iCrate != jCrate || iCard != jCard || iTower != jTower)
		std::cout << "Problem 2:"
			  << "\tiCrate = " << iCrate << "\tiCard  = " << iCard << "\tiTower = " << iTower 
			  << "\tjCrate = " << jCrate << "\tjCard  = " << jCard << "\tjTower = " << jTower 
			  << "\tiEta   = " << iEta << "\tiPhi   = " << iPhi
			  << std::endl;
	    }
	}
    }
  for(unsigned short iCrate = 0; iCrate < 18; iCrate++)
    {
      unsigned short iCard = 999;
      for(unsigned short iTower = 0; iTower < 8; iTower++)
	{
	  short iEta = rctParameters->calcIEta(iCrate, iCard, iTower);
	  short iPhi = rctParameters->calcIPhi(iCrate, iCard, iTower);
	  short jCrate = rctParameters->calcCrate(iPhi, iEta);
	  short jCard = rctParameters->calcCard(iPhi, (unsigned short) abs(iEta));
	  short jTower = rctParameters->calcTower(iPhi, (unsigned short) abs(iEta));
	  if(iCrate != jCrate || iCard != jCard || iTower != jTower)
	    std::cout << "Problem 2:"
		      << "\tiCrate = " << iCrate << "\tiCard  = " << iCard << "\tiTower = " << iTower 
		      << "\tjCrate = " << jCrate << "\tjCard  = " << jCard << "\tjTower = " << jTower 
		      << "\tiEta   = " << iEta << "\tiPhi   = " << iPhi
		      << std::endl;
	}
    }
}
