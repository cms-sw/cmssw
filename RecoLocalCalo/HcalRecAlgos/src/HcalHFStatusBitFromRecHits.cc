#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromRecHits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <algorithm> // for "max"
#include <cmath>
#include <iostream>

HcalHFStatusBitFromRecHits::HcalHFStatusBitFromRecHits()
{
  // use simple values in default constructor
  long_HFlongshortratio_   = 0.99;
  short_HFlongshortratio_  = 0.99;
  long_thresholdET_    = 0.5; // default energy requirement
  short_thresholdET_   = 0.5;
  long_thresholdEnergy_ = 100;
  short_thresholdEnergy_ = 100;
}

HcalHFStatusBitFromRecHits::HcalHFStatusBitFromRecHits(double shortR, double shortET, double shortE,
						       double longR, double longET, double longE)
{
  long_HFlongshortratio_   = longR; 
  short_HFlongshortratio_  = shortR;
  long_thresholdET_        = longET;
  short_thresholdET_       = shortET;
  long_thresholdEnergy_    = longE;
  short_thresholdEnergy_   = shortE;
}

HcalHFStatusBitFromRecHits::~HcalHFStatusBitFromRecHits(){}

void HcalHFStatusBitFromRecHits::hfSetFlagFromRecHits(HFRecHitCollection& rec, 
						      HcalChannelQuality* myqual, 
						      const HcalSeverityLevelComputer* mySeverity)
{
  // Compares energies from long & short fibers
  int status;
  float en, en2;
  int ieta, iphi, depth;
  float ratio; // ratio of (L-S)/(L+S) energy magnitudes
  double coshEta;

  // Is there a faster way to do this than a double loop?
  for (HFRecHitCollection::iterator iHF=rec.begin(); iHF!=rec.end();++iHF)
    {
      // skip cells that have already been tagged -- shouldn't happen in current algorithm
      //if (iHF->flagField(HcalCaloFlagLabels::HFLongShort, HcalCaloFlagLabels::HFLongShort+1)) continue;

      ieta =iHF->id().ieta();  // int between 29-41
      // eta = average value between cell eta bounds
      std::pair<double,double> etas = myqual->topo()->etaRange(HcalForward,abs(ieta));
      double eta1 = etas.first;
      double eta2 = etas.second;
      coshEta=fabs(cosh(0.5*(eta1+eta2)));

      status=0; // status bit for rechit

      en2=-999; // dummy starting value for partner energy
      depth=iHF->id().depth();
      en=iHF->energy(); // energy of current rechit

      if (depth==1) // check long fiber
	{
	  if (en<1.2) continue;  // never flag long rechits < 1.2 GeV
	  if (long_thresholdEnergy_>0. && en<long_thresholdEnergy_) continue;
	  if (long_thresholdET_>0. && en<long_thresholdET_*coshEta) continue;
	}

      else if (depth==2) // check short fiber
	{
	  if (en<1.8) continue;  // never flag short rechits < 1.8 GeV
	  if (short_thresholdEnergy_>0. && en<short_thresholdEnergy_) continue;
	  if (short_thresholdET_>0. && en<short_thresholdET_*coshEta) continue;
	}
      
      iphi =iHF->id().iphi();

      // Check for cells whose partners have been excluded from the rechit collections
      // Such cells will not get flagged (since we can't make an L vs. S comparison)

      HcalDetId partner(HcalForward, ieta, iphi, 3-depth); // if depth=1, 3-depth =2, and vice versa
      DetId detpartner=DetId(partner);
      const HcalChannelStatus* partnerstatus=myqual->getValues(detpartner.rawId());
      if (mySeverity->dropChannel(partnerstatus->getValue() ) ) continue;  // partner was dropped; don't set flag

      // inner loop will find 'partner' channel (same ieta, iphi, different depth)
      for (HFRecHitCollection::iterator iHF2=rec.begin(); iHF2!=rec.end();++iHF2)
	{
	  if (iHF2->id().ieta()!=ieta) continue; // require ieta match
	  if (iHF2->id().iphi()!=iphi) continue; // require iphi match
	  if (iHF2->id().depth()==depth) continue;  // require short/long combo

	  en2=iHF2->energy(); 

	  /* 
	     We used to use absolute values of energies for ratios, but I don't think we want to do this any more.
	     For example, for a threshold of 0.995, if en=50 and en2=0, the flag would be set.
	     But if en=50 and en2<-0.125, the threshold would not be set if using the absolute values.
	     I don't think we want to have a range of en2 below which the flag is not set.
	     This does mean that we need to be careful not to set the en energy threshold too low,
	     so as to not falsely flag fluctuations (en=2, en2=-0.01, for example), but we should never be setting our
	     thresholds that low.
	  */
	  
	  ratio = (en - en2)/(en + en2);
	  
	  if (depth==1 && ratio>long_HFlongshortratio_)
	    status=1;
	  else if (depth==2 && ratio>short_HFlongshortratio_)
	    status=1;
	  break; // once partner channel found, break out of loop
	} // inner loop

      // Consider the case where only one depth present
      if (en2==-999) // outer rechit has already passed energy, ET thresholds above; no partner cell means this looks like isolated energy in one channel -- flag it!
	status=1;

      // flag rechit as potential PMT window hit
      if (status==1) 
	iHF->setFlagField(status,HcalCaloFlagLabels::HFLongShort, 1);
    } // outer loop
  return;
} // void HcalHFStatusBitFromRecHits::hfSetFlagFromRecHits(...)
