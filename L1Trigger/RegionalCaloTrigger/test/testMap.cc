#include <iostream>
#include <string>

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

int main()
{
  std::string filename("../data/TPGcalc.txt");
  L1RCT rct(filename);
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
		iCrate = rct.calcCrate(iPhi, iEta);
		iCard = rct.calcCard(iPhi, (unsigned short) abs(iEta));
		iTower = rct.calcTower(iPhi, (unsigned short) abs(iEta));
		short jEta = rct.calcIEta(iCrate, iCard, iTower);
		unsigned short jPhi = rct.calcIPhi(iCrate, iCard, iTower);
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
	  for(unsigned short iTower = 1; iTower <= 32; iTower++)
	    {
	      short iEta = rct.calcIEta(iCrate, iCard, iTower);
	      unsigned short iPhi = rct.calcIPhi(iCrate, iCard, iTower);
	      short jCrate = rct.calcCrate(iPhi, iEta);
	      short jCard = rct.calcCard(iPhi, (unsigned short) abs(iEta));
	      short jTower = rct.calcTower(iPhi, (unsigned short) abs(iEta));
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
	  short iEta = rct.calcIEta(iCrate, iCard, iTower);
	  short iPhi = rct.calcIPhi(iCrate, iCard, iTower);
	  short jCrate = rct.calcCrate(iPhi, iEta);
	  short jCard = rct.calcCard(iPhi, (unsigned short) abs(iEta));
	  short jTower = rct.calcTower(iPhi, (unsigned short) abs(iEta));
	  unsigned short iAEta = iEta + iAbsEtaMax;
	  if(iCrate != jCrate || iCard != jCard || iTower != jTower)
	    std::cout << "Problem 2:"
		      << "\tiCrate = " << iCrate << "\tiCard  = " << iCard << "\tiTower = " << iTower 
		      << "\tjCrate = " << jCrate << "\tjCard  = " << jCard << "\tjTower = " << jTower 
		      << "\tiEta   = " << iEta << "\tiPhi   = " << iPhi
		      << std::endl;
	}
    }
}
