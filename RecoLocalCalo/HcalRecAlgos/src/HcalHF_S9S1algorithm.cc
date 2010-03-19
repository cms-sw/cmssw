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

using namespace std;

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
      LongSlopes.push_back(CalcSlope(i,blank));
      ShortSlopes.push_back(CalcSlope(i,blank));
    }
  LongEnergyThreshold.clear();
  LongETThreshold.clear();
  ShortEnergyThreshold.clear();
  ShortETThreshold.clear();
  for (int i=29;i<=41;++i)
    {
      LongEnergyThreshold.push_back(CalcEnergyThreshold(1.*i,EnergyDefault));
      LongETThreshold.push_back(CalcEnergyThreshold(1.*i,blank));
      ShortEnergyThreshold.push_back(CalcEnergyThreshold(1.*i,EnergyDefault));
      ShortETThreshold.push_back(CalcEnergyThreshold(1.*i,blank));
    }
}


HcalHF_S9S1algorithm::HcalHF_S9S1algorithm(std::vector<double> short_optimumSlope, 
					 double short_optimumSlope40,
					 double short_optimumSlope41,
					 std::vector<double> short_Energy, 
					 std::vector<double> short_ET, 
					 std::vector<double> long_optimumSlope, 
					 double long_optimumSlope40,
					 double long_optimumSlope41,
					 std::vector<double> long_Energy, 
					 std::vector<double> long_ET)

{
  // Constructor in the case where all parameters are provided by the user

  // Thresholds only need to be computed once, not every event!
  LongSlopes.clear();
  ShortSlopes.clear();
  for (int i=29;i<=39;++i)
    {
      LongSlopes.push_back(CalcSlope(i,long_optimumSlope));
      ShortSlopes.push_back(CalcSlope(i,short_optimumSlope));
    }
  // |ieta|=40 and 41 don't follow polynomial fit to slope, and must be specified separately
  LongSlopes.push_back(long_optimumSlope40);
  LongSlopes.push_back(long_optimumSlope41);
  ShortSlopes.push_back(short_optimumSlope40);
  ShortSlopes.push_back(short_optimumSlope41);

  // Compute energy and ET threshold for long and short fibers at each ieta
  LongEnergyThreshold.clear();
  LongETThreshold.clear();
  ShortEnergyThreshold.clear();
  ShortETThreshold.clear();
  for (int i=29;i<=41;++i)
    {
      LongEnergyThreshold.push_back(CalcEnergyThreshold(1.*i,long_Energy_));
      LongETThreshold.push_back(CalcEnergyThreshold(1.*i,long_ET_));
      ShortEnergyThreshold.push_back(CalcEnergyThreshold(1.*i,short_Energy_));
      ShortETThreshold.push_back(CalcEnergyThreshold(1.*i,short_ET_));
    }
} // HcalHF_S9S1algorithm constructor with parameters

HcalHF_S9S1algorithm::~HcalHF_S9S1algorithm(){}


void HcalHF_S9S1algorithm::HFSetFlagFromS9S1(HFRecHit& hf,
					   HFRecHitCollection& rec,
					   HcalChannelQuality* myqual,
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
    {
      hf.setFlagField(0, HcalCaloFlagLabels::HFLongShort); // shouldn't be necessary, but set bit to 0 just to be sure
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
	  if (i==ieta && d==depth) continue;  // don't add the cell itself
	  // Look to see if neighbor is in rechit collection
	  HcalDetId neighbor(HcalForward, i,testphi,d);
	  HFRecHitCollection::const_iterator neigh=rec.find(neighbor);
	  if (neigh!=rec.end())
	    S9S1+=neigh->energy();
	}
    }

  // Part B: Fix ieta, and loop over iphi.  A bit more tricky, because of iphi wraparound and different segmentation at 40, 41
  
  int phiseg=2; // 10 degree segmentation for most of HF (1 iphi unit = 5 degrees)
  if (abs(ieta)>39) phiseg=4; // 20 degree segmentation for |ieta|>39
  for (int d=1;d<=2;++d)
    {
      for (int i=iphi-phiseg;i<=iphi+phiseg;i+=phiseg)
	{
	  testphi=i;
	  // Our own modular function, since default produces results -1%72 = -1
	  while (testphi<0) testphi+=72;
	  while (testphi>72) testphi-=72;
	  if (testphi==iphi && d==depth) continue;  // don't add the cell itself
	  // Look to see if neighbor is in rechit collection
	  HcalDetId neighbor(HcalForward, ieta,testphi,d);
	  HFRecHitCollection::const_iterator neigh=rec.find(neighbor);
	  if (neigh!=rec.end())
	    S9S1+=neigh->energy();
	}
    }
  
  if (abs(ieta)==40) // add extra cell for 39/40 boundary due to increased phi size at ieta=40.
    {
      HcalDetId neighbor(HcalForward, 39*abs(ieta)/ieta,(iphi+2)%72,depth);  
      HFRecHitCollection::const_iterator neigh=rec.find(neighbor);
      if (neigh!=rec.end())
	S9S1+=neigh->energy();
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
  if (S9S1>=S9S1cut)
    hf.setFlagField(0,HcalCaloFlagLabels::HFLongShort); // doesn't look like noise
  else
    hf.setFlagField(1,HcalCaloFlagLabels::HFLongShort);
  return;
} // void HcalHF_S9S1algorithm::HFSetFlagFromS9S1



double HcalHF_S9S1algorithm::CalcSlope(int abs_ieta, std::vector<double> params)
{
  /* CalcIetaThreshold calculates the polynomial [0]+[1]*x + [2]*x^2 + ....,
     where x is an integer provided by the first argument (int abs_ieta),
     and [0],[1],[2] is a vector of doubles provided by the second (std::vector<double> params).
     The output of the polynomial calculation (threshold) is returned by the function.
  */
  double threshold=0;
  for (std::vector<double>::size_type i=0;i<params.size();++i)
    {
      threshold+=params[i]*pow(abs_ieta, (int)i);
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

