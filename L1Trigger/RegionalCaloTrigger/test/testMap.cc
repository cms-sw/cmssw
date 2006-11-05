#include <iostream>
#include <math.h>

short calcIEta(unsigned short iCrate, unsigned short iCard, unsigned short iTower)
{
  unsigned short absIEta;
  if(iCard < 6) absIEta = (iCard % 3) * 8 + ((iTower - 1) / 4) + 1;
  if(iCard == 6) {
    if(iTower < 17)
      absIEta = ((iTower - 1) / 4) + 25;
    else
      absIEta = ((iTower - 17) / 4) + 25;
  }
  short iEta;
  if(iCrate < 9) iEta = -absIEta;
  else iEta = absIEta;
  return iEta;
}

short calcIPhi(unsigned short iCrate, unsigned short iCard, unsigned short iTower)
{
  short iPhi;
  if(iCard < 6)
    iPhi = (iCrate % 9) * 8 + (iCard / 3) * 4 + ((iTower - 1) % 4);
  else {
    if(iTower < 17)
      iPhi = (iCrate % 9) * 8 + ((iTower - 1) % 4);
    else
      iPhi = (iCrate % 9) * 8 + ((iTower - 17) % 4) + 4;
  }
}

// maps rct iphi, ieta of tower to crate
unsigned short calcCrate(unsigned short rct_iphi, short ieta){
  unsigned short crate = rct_iphi/8;
  if (ieta > 0){
    crate = crate + 9;
  }
  return crate;
}

//map digi rct iphi, ieta to card
unsigned short calcCard(unsigned short rct_iphi, unsigned short absIeta){
  unsigned short card = 999;
  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    card =  ((absIeta-1)/8) + ((rct_iphi / 4) % 2) * 3 ;          // integer division again
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    card = 6;
  }
  else{}
  return card;
}

//map digi rct iphi, ieta to tower
unsigned short calcTower(unsigned short rct_iphi, unsigned short absIeta){
  unsigned short tower = 999;
  unsigned short iphi = rct_iphi;
  unsigned short regionPhi = (iphi % 8)/4;

  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    tower = ((absIeta-1)%8)*4 + (iphi%4) + 1;       // assume iphi between 0 and 71; makes towers from 1-32
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    if (regionPhi == 0){
      tower = (absIeta-25)*4 + (iphi%4) + 1;   // towers from 1-32, modified Aug. 1 Jessica Leonard
    }
    else {
      tower = (absIeta-25)*4 + (iphi%4) + 17;
    }
  }
  // absIeta >= 29 (HF regions)
  else if ((absIeta >= 29) && (absIeta <= 32)){
    regionPhi = iphi % 2;  // SPECIAL DEFINITION OF REGIONPHI FOR HF SINCE HF IPHI IS 0-17 Sept. 19 J. Leonard
    // HF MAPPING, just regions now, don't need to worry about towers -- just calling it "tower" for convenience
    tower = (regionPhi) * 4 + absIeta - 29;
  }
  return tower;
}

int main()
{
  unsigned short iPhiMax = 72;
  short iAbsEtaMax = 28;
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
	      crate[iPhi][iAEta] = calcCrate(iPhi, iEta);
	      card[iPhi][iAEta] = calcCard(iPhi, (unsigned short) abs(iEta));
	      tower[iPhi][iAEta] = calcTower(iPhi, (unsigned short) abs(iEta));
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
	      short iEta = calcIEta(iCrate, iCard, iTower);
	      short iPhi = calcIPhi(iCrate, iCard, iTower);
	      short jCrate = calcCrate(iPhi, iEta);
	      short jCard = calcCard(iPhi, (unsigned short) abs(iEta));
	      short jTower = calcTower(iPhi, (unsigned short) abs(iEta));
	      short jEta = calcIEta(jCrate, jCard, jTower);
	      short jPhi = calcIPhi(jCrate, jCard, jTower);
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
}

