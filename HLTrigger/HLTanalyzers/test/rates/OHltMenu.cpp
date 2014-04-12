#define OHltMenu_cxx
#include "OHltMenu.h"

void OHltMenu::AddHlt(TString trig, TString l1Bit, int l1prescale, int hltprescale,
		      TString threshold, TString desc)
{
  hlts.push_back(trig);
  hltL1Bit[trig] 	       	= l1Bit;
  hltThreshold[trig] 		= threshold;
  hltDescription[trig] 		= desc;
  l1Prescale[trig]		= l1prescale;
  hltPrescale[trig]		= hltprescale;

  int tmpL1prescale = 1;
  int tmpHLTprescale = 1;
  if (l1prescale>0) tmpL1prescale   = l1prescale;   // if <=0: prescale already applied on trigger
  if (hltprescale>0) tmpHLTprescale = hltprescale;  // if <=0: prescale already applied on trigger
  totalPrescale[trig]		= tmpL1prescale * tmpHLTprescale;
}

void OHltMenu::AddHlt(TString trig, TString l1Bit, int hltprescale,
		      TString threshold, TString desc)
{
  hlts.push_back(trig);
  hltL1Bit[trig] 	       	= l1Bit;
  hltThreshold[trig] 		= threshold;
  hltDescription[trig] 		= desc;
  l1Prescale[trig]		= 1;
  hltPrescale[trig]		= hltprescale;
  totalPrescale[trig]		= hltprescale;
}
