#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_S9S1algorithm.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h" // for eta bounds
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

#include <algorithm> // for "max"
#include <cmath>
#include <iostream>

HcalHF_S9S1algorithm::HcalHF_S9S1algorithm()
{ 
  // Default settings:  Energy > 50 GeV, slope = 0, ET = 0
  std::vector<double> blank;
  blank.clear();
  blank.push_back(0);
  std::vector<double> EnergyDefault;
  EnergyDefault.clear();
  EnergyDefault.push_back(50);

  // Thresholds only need to be computed once, not every event!
  LongSlopes.clear();
  ShortSlopes.clear();
  for (int i=29;i<=41;++i)
    {
      LongSlopes.push_back(0);
      ShortSlopes.push_back(0);
    }
  LongEnergyThreshold.clear();
  LongETThreshold.clear();
  ShortEnergyThreshold.clear();
  ShortETThreshold.clear();
  for (int i=29;i<=41;++i)
    {
      LongEnergyThreshold.push_back(EnergyDefault[0]);
      LongETThreshold.push_back(blank[0]);
      ShortEnergyThreshold.push_back(EnergyDefault[0]);
      ShortETThreshold.push_back(blank[0]);
    }
  HcalAcceptSeverityLevel_=0;
  isS8S1_=false; // S8S1 is almost the same as S9S1
}


HcalHF_S9S1algorithm::HcalHF_S9S1algorithm(std::vector<double> short_optimumSlope, 
					   std::vector<double> short_Energy, 
					   std::vector<double> short_ET, 
					   std::vector<double> long_optimumSlope, 
					   std::vector<double> long_Energy, 
					   std::vector<double> long_ET,
					   int HcalAcceptSeverityLevel,
					   bool isS8S1)

{
  // Constructor in the case where all parameters are provided by the user

  // Thresholds only need to be computed once, not every event!

  LongSlopes=long_optimumSlope;
  ShortSlopes=short_optimumSlope;
  
  while (LongSlopes.size()<13)
    LongSlopes.push_back(0); // should be unnecessary, but include this protection to avoid crashes
  while (ShortSlopes.size()<13)
    ShortSlopes.push_back(0);

  // Get long, short energy thresholds (different threshold for each |ieta|)
  LongEnergyThreshold.clear();
  LongETThreshold.clear();
  ShortEnergyThreshold.clear();
  ShortETThreshold.clear();
  LongEnergyThreshold=long_Energy;
  LongETThreshold=long_ET;
  ShortEnergyThreshold=short_Energy;
  ShortETThreshold=short_ET;

  HcalAcceptSeverityLevel_=HcalAcceptSeverityLevel;
  isS8S1_=isS8S1;
} // HcalHF_S9S1algorithm constructor with parameters

HcalHF_S9S1algorithm::~HcalHF_S9S1algorithm(){}


void HcalHF_S9S1algorithm::HFSetFlagFromS9S1(HFRecHit& hf,
					     HFRecHitCollection& rec,
					     const HcalChannelQuality* myqual,
					     const HcalSeverityLevelComputer* mySeverity)

{
  int ieta=hf.id().ieta(); // get coordinates of rechit being checked
  int depth=hf.id().depth();
  int iphi=hf.id().iphi();
  double fEta = 0.5*(theHFEtaBounds[abs(ieta)-29] + theHFEtaBounds[abs(ieta)-28]); // calculate eta as average of eta values at ieta boundaries
  double energy=hf.energy();
  double ET = energy/fabs(cosh(fEta));

  // Step 1:  Check eta-dependent energy and ET thresholds -- same as PET algorithm
  double ETthresh=0, Energythresh=0; // set ET, energy thresholds
  if (depth==1)  // set thresholds for long fibers
    {
      Energythresh = LongEnergyThreshold[abs(ieta)-29];
      ETthresh     = LongETThreshold[abs(ieta)-29];
    }
  else if (depth==2) // short fibers
    {
      Energythresh = ShortEnergyThreshold[abs(ieta)-29];
      ETthresh     = ShortETThreshold[abs(ieta)-29];
    }
  if (energy<Energythresh || ET < ETthresh)
    return;

  // Step 1A:
  // Check that EL<ES when evaluating short fibers  (S8S1 check only)
  if (depth==2 && abs(ieta)>29 && isS8S1_)
    {
      double EL=0;
      // look for long partner
      HcalDetId neighbor(HcalForward, ieta,iphi,1);
      HFRecHitCollection::const_iterator neigh=rec.find(neighbor);
      if (neigh!=rec.end())
	EL=neigh->energy();
      
      if (EL>=energy)
	return;
    }

  // Step 2:  Find all neighbors, and calculate S9/S1
  double S9S1=0;
  int testphi=-99;

  // Part A:  Check fixed iphi, and vary ieta
  for (int d=1;d<=2;++d) // depth loop
    {
      for (int i=ieta-1;i<=ieta+1;++i) // ieta loop
	{
	  testphi=iphi;
	  // Special case when ieta=39, since ieta=40 only has phi values at 3,7,11,...
	  // phi=3 covers 3,4,5,6
	  if (abs(ieta)==39 && abs(i)>39 && testphi%4==1)
	    testphi-=2;
	  while (testphi<0) testphi+=72;
	  if (i==ieta)
	    if (d==depth || isS8S1_==true) continue;  // don't add the cell itself; don't count neighbor in same ieta-phi if S8S1 test enabled

	  // Look to see if neighbor is in rechit collection
	  HcalDetId neighbor(HcalForward, i,testphi,d);
	  HFRecHitCollection::const_iterator neigh=rec.find(neighbor);
	  // require that neighbor exists, and that it doesn't have a prior flag already set
	  if (neigh!=rec.end())
	    {
	      const uint32_t chanstat = myqual->getValues(neighbor)->getValue();
	      int SeverityLevel=mySeverity->getSeverityLevel(neighbor, neigh->flags(),
							     chanstat);
	      if (SeverityLevel<=HcalAcceptSeverityLevel_)
		S9S1+=neigh->energy();
	    }
	}
    }

  // Part B: Fix ieta, and loop over iphi.  A bit more tricky, because of iphi wraparound and different segmentation at 40, 41
  
  int phiseg=2; // 10 degree segmentation for most of HF (1 iphi unit = 5 degrees)
  if (abs(ieta)>39) phiseg=4; // 20 degree segmentation for |ieta|>39
  for (int d=1;d<=2;++d)
    {
      for (int i=iphi-phiseg;i<=iphi+phiseg;i+=phiseg)
	{
	  if (i==iphi) continue;  // don't add the cell itself, or its depthwise partner (which is already counted above)
	  testphi=i;
	  // Our own modular function, since default produces results -1%72 = -1
	  while (testphi<0) testphi+=72;
	  while (testphi>72) testphi-=72;
	  // Look to see if neighbor is in rechit collection
	  HcalDetId neighbor(HcalForward, ieta,testphi,d);
	  HFRecHitCollection::const_iterator neigh=rec.find(neighbor);
	  if (neigh!=rec.end())
	    {
              const uint32_t chanstat = myqual->getValues(neighbor)->getValue();
              int SeverityLevel=mySeverity->getSeverityLevel(neighbor, neigh->flags(),
                                                             chanstat);
              if (SeverityLevel<=HcalAcceptSeverityLevel_)
                S9S1+=neigh->energy();
            }
	}
    }
  
  if (abs(ieta)==40) // add extra cells for 39/40 boundary due to increased phi size at ieta=40.
    {
      for (int d=1;d<=2;++d) // add cells from both depths!
	{
	  HcalDetId neighbor(HcalForward, 39*abs(ieta)/ieta,(iphi+2)%72,d);  
	  HFRecHitCollection::const_iterator neigh=rec.find(neighbor);
	  if (neigh!=rec.end())
            {
              const uint32_t chanstat = myqual->getValues(neighbor)->getValue();
              int SeverityLevel=mySeverity->getSeverityLevel(neighbor, neigh->flags(),
                                                             chanstat);
              if (SeverityLevel<=HcalAcceptSeverityLevel_)
                S9S1+=neigh->energy();
            }

	}
    }
    
  // So far, S9S1 is the sum of the neighbors; divide to form ratio
  S9S1/=energy;

  // Now compare to threshold
  double slope=0;
  if (depth==1) slope = LongSlopes[abs(ieta)-29];
  else if (depth==2) slope=ShortSlopes[abs(ieta)-29];
  double intercept = 0;
  if (depth==1) intercept = LongEnergyThreshold[abs(ieta)-29];
  else if (depth==2)  intercept = ShortEnergyThreshold[abs(ieta)-29];

  // S9S1 cut has the form [0] + [1]*log[E];  S9S1 value should be above this line
  double S9S1cut = 0;
  // Protection in case intercept or energy are ever less than 0.  Do we have some other default value of S9S1cut we'd like touse in this case?
  if (intercept>0 && energy>0)  
    S9S1cut=-1.*slope*log(intercept) + slope*log(energy);
  if (S9S1<S9S1cut)
    {
      // Only set HFS8S1Ratio if S8/S1 ratio test fails
      if (isS8S1_==true)
	hf.setFlagField(1,HcalCaloFlagLabels::HFS8S1Ratio);
      // *Always* set the HFLongShort bit if either S8S1 or S9S1 fail
      hf.setFlagField(1,HcalCaloFlagLabels::HFLongShort);
    }
  return;
} // void HcalHF_S9S1algorithm::HFSetFlagFromS9S1



double HcalHF_S9S1algorithm::CalcSlope(int abs_ieta, std::vector<double> params)
{
  /* CalcSlope calculates the polynomial [0]+[1]*x + [2]*x^2 + ....,
     where x is an integer provided by the first argument (int abs_ieta),
     and [0],[1],[2] is a vector of doubles provided by the second (std::vector<double> params).
     The output of the polynomial calculation (threshold) is returned by the function.
     This function should no longer be needed, since we pass slopes for all ietas into the function via the parameter set.
  */
  double threshold=0;
  for (std::vector<double>::size_type i=0;i<params.size();++i)
    {
      threshold+=params[i]*pow(static_cast<double>(abs_ieta), (int)i);
    }
  return threshold;
} // HcalHF_S9S1algorithm::CalcRThreshold(int abs_ieta, std::vector<double> params)
  



double HcalHF_S9S1algorithm::CalcEnergyThreshold(double abs_energy,std::vector<double> params)
{
  /* CalcEnergyThreshold calculates the polynomial [0]+[1]*x + [2]*x^2 + ....,
     where x is an integer provided by the first argument (int abs_ieta),
     and [0],[1],[2] is a vector of doubles provided by the second (std::vector<double> params).
     The output of the polynomial calculation (threshold) is returned by the function.
  */
  double threshold=0;
  for (std::vector<double>::size_type i=0;i<params.size();++i)
    {
      threshold+=params[i]*pow(abs_energy, (int)i);
    }
  return threshold;
} //double HcalHF_S9S1algorithm::CalcEnergyThreshold(double abs_energy,std::vector<double> params)

