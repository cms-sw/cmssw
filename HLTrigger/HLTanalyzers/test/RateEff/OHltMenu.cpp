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
  l1Prescale[trig]		= "1";
  hltPrescale[trig]		= hltprescale;
  totalPrescale[trig]		= hltprescale;
}

void OHltMenu::AddHlt(TString trig, TString l1Bit, TString l1prescale, int hltprescale,
		      TString threshold, TString desc, double eventsize, int multele, int multpho, int multmu, int multjets, int multmet)
{
  hlts.push_back(trig);
  hltL1Bit[trig] 	       	= l1Bit;
  hltThreshold[trig] 		= threshold;
  hltDescription[trig]          = desc;
  l1Prescale[trig]		= l1prescale;
  hltPrescale[trig]		= hltprescale;
  hltmulele[trig]               = multele;
  hltmulpho[trig]               = multpho;
  hltmulmu[trig]                = multmu;
  hltmuljets[trig]              = multjets;
  hltmulmet[trig]               = multmet;

  TString tmpL1prescale = "1";
  int tmpHLTprescale = 1;
  if (hltprescale>0) tmpHLTprescale = hltprescale;  // if <=0: prescale already applied on trigger
  totalPrescale[trig]               = tmpHLTprescale;
  eventSize[trig]               = eventsize;
}

void OHltMenu::AddHlt(TString trig, TString l1Bit, int l1prescale, int hltprescale,
		      TString threshold, TString desc, double eventsize, int multele, int multpho, int multmu, int multjets, int multmet)
{
  hlts.push_back(trig);
  hltL1Bit[trig]        = l1Bit;
  hltThreshold[trig] = threshold;
  hltDescription[trig]          = desc;
  l1Prescale[trig]= "";
  hltPrescale[trig]= hltprescale;
  hltmulele[trig]               = multele;
  hltmulpho[trig]               = multpho;
  hltmulmu[trig]                = multmu;
  hltmuljets[trig]              = multjets;
  hltmulmet[trig]               = multmet;

  int tmpL1prescale = 1;
  int tmpHLTprescale = 1;
  if (l1prescale>0) tmpL1prescale   = l1prescale;   // if <=0: prescale already applied on trigger
  if (hltprescale>0) tmpHLTprescale = hltprescale;  // if <=0: prescale already applied on trigger
  totalPrescale[trig]= tmpL1prescale * tmpHLTprescale;
  eventSize[trig]               = eventsize;
}

void OHltMenu::AddL1(TString trig, int l1prescale)
{
  levelones.push_back(trig);
  int tmpL1prescale = 1;
  if(l1prescale>0) tmpL1prescale = l1prescale;
  levelonePrescale[trig] = tmpL1prescale; 

  // Initialize a counter for all passing L1 events
  unprescaledl1counter[trig] = 0;
}
