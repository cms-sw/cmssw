#include <iostream>
#include <string>

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

int main()
{
  std::string filename("../data/TPGcalc.txt");
  L1RCT rct(filename);
  const unsigned short iPhiMax = 72;
  const short iAbsEtaMax = 32;
  unsigned short crate[iPhiMax][2*iAbsEtaMax+1];
  unsigned short card[iPhiMax][2*iAbsEtaMax+1];
  unsigned short tower[iPhiMax][2*iAbsEtaMax+1];
  for(unsigned short iPhi = 0; iPhi < iPhiMax; iPhi++)
    {
      for(short iEta = -iAbsEtaMax; iEta <= iAbsEtaMax; iEta++)
	{
	  unsigned iAEta = iEta + iAbsEtaMax;
	  if(iEta != 0)
	    {
	      crate[iPhi][iAEta] = rct.calcCrate(iPhi, iEta);
	      card[iPhi][iAEta] = rct.calcCard(iPhi, (unsigned short) abs(iEta));
	      tower[iPhi][iAEta] = rct.calcTower(iPhi, (unsigned short) abs(iEta));
	    }
	  else
	    {
	      crate[iPhi][iAEta] = 999;
	      tower[iPhi][iAEta] = 999;
	      tower[iPhi][iAEta] = 999;
	    }
	}
    }
  for(unsigned short iCrate = 0; iCrate < 18; iCrate++)
    {
      for(unsigned short iCard = 0; iCard < 7; iCard++)
	{
	  for(unsigned short iTower = 1; iTower <= 32; iTower++)
	    {
	      short iEta = rct.calcIEta(iCrate, iCard, iTower);
	      short iPhi = rct.calcIPhi(iCrate, iCard, iTower);
	      short jCrate = rct.calcCrate(iPhi, iEta);
	      short jCard = rct.calcCard(iPhi, (unsigned short) abs(iEta));
	      short jTower = rct.calcTower(iPhi, (unsigned short) abs(iEta));
	      short jEta = rct.calcIEta(jCrate, jCard, jTower);
	      short jPhi = rct.calcIPhi(jCrate, jCard, jTower);
	      unsigned short iAEta = iEta + iAbsEtaMax;
	      if(iCrate != crate[iPhi][iAEta] || iCard != card[iPhi][iAEta] || iTower != tower[iPhi][iAEta])
		std::cout << "Problem:"
			  << "\tiCrate = " << iCrate << "\tcrate[][] = " << crate[iPhi][iAEta] << "\tjCrate = " << jCrate 
			  << "\tiCard  = " << iCard  << "\tcard[][]  = " << card[iPhi][iAEta]  << "\tjCard  = " << jCard  
			  << "\tiTower = " << iTower << "\ttower[][] = " << tower[iPhi][iAEta] << "\tjTower = " << jTower 
			  << "\tiEta   = " << iEta << "\tjEta   = " << jEta
			  << "\tiPhi   = " << iPhi << "\tjPhi   = " << jPhi
			  << std::endl;
	    }
	}
    }
  for(unsigned short iCrate = 0; iCrate < 18; iCrate++)
    {
      unsigned short iCard = 999;
      for(unsigned short iTower = 0; iTower < 8; iTower++)
	{
	  short iEta = rct.calcIEta(iCrate, iCard, iTower);
	  short iPhi = rct.calcIPhi(iCrate, iCard, iTower);
	  short jCrate = rct.calcCrate(iPhi, iEta);
	  short jCard = rct.calcCard(iPhi, (unsigned short) abs(iEta));
	  short jTower = rct.calcTower(iPhi, (unsigned short) abs(iEta));
	  short jEta = rct.calcIEta(jCrate, jCard, jTower);
	  short jPhi = rct.calcIPhi(jCrate, jCard, jTower);
	  unsigned short iAEta = iEta + iAbsEtaMax;
	  if(iCrate != crate[iPhi][iAEta] || iCard != card[iPhi][iAEta] || iTower != tower[iPhi][iAEta])
	    std::cout << "Problem:"
		      << "\tiCrate = " << iCrate << "\tcrate[][] = " << crate[iPhi][iAEta] << "\tjCrate = " << jCrate 
		      << "\tiCard  = " << iCard  << "\tcard[][]  = " << card[iPhi][iAEta]  << "\tjCard  = " << jCard  
		      << "\tiTower = " << iTower << "\ttower[][] = " << tower[iPhi][iAEta] << "\tjTower = " << jTower 
		      << "\tiEta   = " << iEta << "\tjEta   = " << jEta
		      << "\tiPhi   = " << iPhi << "\tjPhi   = " << jPhi
		      << std::endl;
	}
    }
}

