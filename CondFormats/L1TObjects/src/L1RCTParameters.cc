/**
 * Author: Sridhara Dasu
 * Created: 04 July 2007
 * $Id: L1RCTParameters.cc,v 1.5 2007/07/26 09:07:53 dasu Exp $
 **/

#include <iostream>
#include <fstream>

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1RCTParameters::L1RCTParameters(double eGammaLSB,
				 double jetMETLSB,
				 double eMinForFGCut,
				 double eMaxForFGCut,
				 double hOeCut,
				 double eMinForHoECut,
				 double eMaxForHoECut,
				 double eActivityCut,
				 double hActivityCut
				 ) :
  eGammaLSB_(eGammaLSB),
  jetMETLSB_(jetMETLSB),
  eMinForFGCut_(eMinForFGCut),
  eMaxForFGCut_(eMaxForFGCut),
  hOeCut_(hOeCut),
  eMinForHoECut_(eMinForHoECut),
  eMaxForHoECut_(eMaxForHoECut),
  eActivityCut_(eActivityCut),
  hActivityCut_(hActivityCut)
{
}

// maps rct iphi, ieta of tower to crate
unsigned short L1RCTParameters::calcCrate(unsigned short rct_iphi, short ieta) const
{
  unsigned short crate = rct_iphi/8;
  if(abs(ieta) > 28) crate = rct_iphi / 2;
  if (ieta > 0){
    crate = crate + 9;
  }
  return crate;
}

//map digi rct iphi, ieta to card
unsigned short L1RCTParameters::calcCard(unsigned short rct_iphi, 
					 unsigned short absIeta) const
{
  unsigned short card = 999;
  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    card =  ((absIeta-1)/8)*2 + (rct_iphi%8)/4;
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    card = 6;
  }
  else{}
  return card;
}

//map digi rct iphi, ieta to tower
unsigned short L1RCTParameters::calcTower(unsigned short rct_iphi, 
					  unsigned short absIeta) const
{
  unsigned short tower = 999;
  unsigned short iphi = rct_iphi;
  unsigned short regionPhi = (iphi % 8)/4;

  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    // assume iphi between 0 and 71; makes towers from 1-32
    tower = ((absIeta-1)%8)*4 + (iphi%4) + 1;
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    if (regionPhi == 0){
      // towers from 1-32, modified Aug. 1 Jessica Leonard
      tower = (absIeta-25)*4 + (iphi%4) + 1;
    }
    else {
      tower = 29 + iphi % 4 + (25 - absIeta) * 4;
    }
  }
  // absIeta >= 29 (HF regions)
  else if ((absIeta >= 29) && (absIeta <= 32)){
    // SPECIAL DEFINITION OF REGIONPHI FOR HF SINCE HF IPHI IS 0-17 
    // Sept. 19 J. Leonard
    regionPhi = iphi % 2;
    // HF MAPPING, just regions now, don't need to worry about towers
    // just calling it "tower" for convenience
    tower = (regionPhi) * 4 + absIeta - 29;
  }
  return tower;
}

short L1RCTParameters::calcIEta(unsigned short iCrate, unsigned short iCard, 
				unsigned short iTower) const
{
  unsigned short absIEta;
  if(iCard < 6) 
    absIEta = (iCard / 2) * 8 + ((iTower - 1) / 4) + 1;
  else if(iCard == 6) {
    if(iTower < 17)
      absIEta = 25 + (iTower - 1) / 4;
    else
      absIEta = 28 - ((iTower - 17) / 4);
  }
  else
    absIEta = 29 + iTower % 4;
  short iEta;
  if(iCrate < 9) iEta = -absIEta;
  else iEta = absIEta;
  return iEta;
}

unsigned short L1RCTParameters::calcIPhi(unsigned short iCrate, 
					 unsigned short iCard, 
					 unsigned short iTower) const
{
  short iPhi;
  if(iCard < 6)
    iPhi = (iCrate % 9) * 8 + (iCard % 2) * 4 + ((iTower - 1) % 4);
  else if(iCard == 6){
    if(iTower < 17)
      iPhi = (iCrate % 9) * 8 + ((iTower - 1) % 4);
    else
      iPhi = (iCrate % 9) * 8 + ((iTower - 17) % 4) + 4;
  }
  else
    iPhi = (iCrate % 9) * 2 + iTower / 4;
  return iPhi;
}
