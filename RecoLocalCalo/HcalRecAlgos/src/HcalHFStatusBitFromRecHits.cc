#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromRecHits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h" // for eta bounds

#include <algorithm> // for "max"
#include <cmath>
#include <iostream>
using namespace std;

HcalHFStatusBitFromRecHits::HcalHFStatusBitFromRecHits()
{
  // use simple values in default constructor
  HFlongshortratio_   = .99;
  thresholdET_    = 0.5; // default energy requirement
}

HcalHFStatusBitFromRecHits::HcalHFStatusBitFromRecHits(double HFlongshort,  double thresholdET)
{
  HFlongshortratio_   = HFlongshort;
  thresholdET_    = thresholdET;
}

HcalHFStatusBitFromRecHits::~HcalHFStatusBitFromRecHits(){}

void HcalHFStatusBitFromRecHits::hfSetFlagFromRecHits(HFRecHitCollection& rec)
{
  // Compares energies from long & short fibers
  int status;
  float enL, enS;
  int ieta, iphi, depth;
  float ratio; // ratio of (L-S)/(L+S) energy magnitudes
  double coshEta;
  // Probably a faster way to do this than a double loop?
  for (HFRecHitCollection::iterator iHF=rec.begin(); iHF!=rec.end();++iHF)
    {
      // skip cells that have already been tagged
      if (iHF->flagField(HcalCaloFlagLabels::HFLongShort, HcalCaloFlagLabels::HFLongShort+1)) continue;
      /* 
	 Don't run noise algorithm on cells with ET (energy?) < threshold
	 If energy> threshold and noise algorithm passed, mark both long and short fibers. 
      */
      ieta =iHF->id().ieta();  // int between 29-41
      coshEta=fabs(cosh(0.5*(theHFEtaBounds[abs(ieta)-29]+theHFEtaBounds[abs(ieta)-28])));
      // requre ET> thresholdET_; ETcosh(eta) = E;
      if (iHF->energy()<thresholdET_*coshEta) continue;
      status=0; // status bit for rechit
      enL=-999; // dummy starting values for long, short fiber energies
      enS=-999;
      depth=iHF->id().depth();
      
      if (depth==1) 
	enL=iHF->energy();
      else 
	enS =iHF->energy();
      
      iphi =iHF->id().iphi();
      
      for (HFRecHitCollection::iterator iHF2=rec.begin(); iHF2!=rec.end();++iHF2)
	{
	  if ((iHF2->id().depth()+depth)!=3) continue; // require short/long combo
	  if (iHF2->id().ieta()!=ieta) continue; // require ieta match
	  if (iHF2->id().iphi()!=iphi) continue; // require iphi match
	  if (iHF2->id().depth()==1)
	    enL=iHF2->energy();
	  else
	    enS=iHF2->energy();

	  if (enS<1.8 && enL<1.2) break; // energy too low for both; skip 
	  ratio = ((fabs)(enL) - fabs(enS))/((fabs)(enL)+(fabs)(enS));
	  
	  if (fabs(ratio)>HFlongshortratio_)
	    {
	      status=1;
	      // set flags for both long and short fiber
	      iHF->setFlagField(status,HcalCaloFlagLabels::HFLongShort, 1);
              iHF2->setFlagField(status,HcalCaloFlagLabels::HFLongShort, 1);
	    }	
	  break;
	} // inner loop
      // Consider the case where only one depth present
      if (enS==-999 && enL>=thresholdET_*coshEta) status=1;
      else if (enL==-999 && enS>=thresholdET_*coshEta) status=1;
      if (status==1) 
	iHF->setFlagField(status,HcalCaloFlagLabels::HFLongShort, 1);
    } // outer loop
  return;
} // void HcalHFStatusBitFromRecHits::hfSetFlagFromRecHits(...)
