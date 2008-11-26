#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromRecHits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm> // for "max"
#include <math.h>
#include <iostream>

HcalHFStatusBitFromRecHits::HcalHFStatusBitFromRecHits()
{
  // use simple values in default constructor
  HFlongshortratio_   = .99;
  bit_=1;
}

HcalHFStatusBitFromRecHits::HcalHFStatusBitFromRecHits(double HFlongshort, int bit)
{
  HFlongshortratio_   = HFlongshort;
  bit_=bit;
}

HcalHFStatusBitFromRecHits::~HcalHFStatusBitFromRecHits(){}

void HcalHFStatusBitFromRecHits::hfSetFlagFromRecHits(HFRecHitCollection& rec)
{
  // Compares energies from long & short fibers
  int status;
  float enL, enS;
  int ieta, iphi, depth;
  // Probably a faster way to do this than a double loop?
  for (HFRecHitCollection::iterator iHF=rec.begin(); iHF!=rec.end();++iHF)
    {
      if ((iHF->flags()&0x2)==1) continue;
      status=0;
      enL=0;
      enS=0;
      depth=iHF->id().depth();
      if (depth==1) enL=iHF->energy();
      else enS =iHF->energy();
      ieta =iHF->id().ieta();
      iphi =iHF->id().iphi();
      
      for (HFRecHitCollection::iterator iHF2=rec.begin(); iHF2!=rec.end();++iHF2)
	{
	  if ((iHF2->id().depth()+depth)!=3) continue; // require short/long combo
	  if (iHF2->id().ieta()!=ieta) continue;
	  if (iHF2->id().iphi()!=iphi) continue;
	  if (iHF2->id().depth()==1)
	    enL=iHF2->energy();
	  else
	    enS=iHF2->energy();
	  if (enS<1.8 && enL<1.2) break; // energy too low for both; skip 
	  if (enS+enL==0) break; // don't yet know how to properly handle this case
	  if (fabs((enL-enS)/(enL+enS))>HFlongshortratio_)
	    {
	      status=1;
	      iHF->setFlags(iHF->flags()|(status<<1));
	      iHF2->setFlags(iHF2->flags()|(status<<1));
	    }	
	  break;
	} // inner loop
      // Consider the case where only one depth present
      if (enS==0 && enL>20) status=1;
      else if (enL==0 && enL>20) status=1;
      if (status==1) 
	iHF->setFlags(iHF->flags()|(status<<bit_));
    } // outer loop
  return;
} // void HcalHFStatusBitFromRecHits::hfSetFlagFromRecHits(...)
