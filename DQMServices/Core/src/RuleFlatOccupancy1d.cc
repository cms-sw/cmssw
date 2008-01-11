#include "DQMServices/Core/interface/QStatisticalTests.h"
#include "DQMServices/Core/interface/RuleFlatOccupancy1d.h"
#include <iostream>

using namespace std;

float RuleFlatOccupancy1d::runTest( const TH1F* const histogram )
{
/*--------------------------------------------------------------------------+
 |                 Input to this function                                   |
 +--------------------------------------------------------------------------+
 |TH1F* histogram,      : histogram to be compared with Rule                |
 |int* mask             : bit mask which excludes bins from consideration   |
 |double epsilon_min,   : minimum tolerance (fraction of line)              |
 |double epsilon_max,   : maximum tolerance (fraction of line)              |
 |double S_fail,        : required Statistical Significance to fail rule    |
 |double S_pass,        : required Significance to pass rule                |
 |double[2][] FailedBins: uninit. vector of bins out of tolerance           | 
 +--------------------------------------------------------------------------+
 |                 Result values for this function                          |
 +--------------------------------------------------------------------------+
 |int result            : "0" = "Passed Rule & is statistically significant"|
 |                        "1" = "Failed Rule & is statistically significant"|
 |                        "2" = "Passed Rule & not stat. significant"       |
 |                        "3" = "Failed Rule & not stat. significant"       |
 |                        "4" = "zero histo entries, can not evaluate Rule" |
 |                        "5" = "Input invalid,      can not evaluate Rule" |
 |double[2][] FailedBins: the obs. vector of bins out of tolerance          | 
 +--------------------------------------------------------------------------+
 | Author: Richard Cavanaugh, University of Florida                         |
 | email:  Richard.Cavanaugh@cern.ch                                        |
 | Creation Date: 07.Jan.2006                                               |
 | Last Modified: 16.Jan.2006                                               |
 | Comments:                                                                |
  +--------------------------------------------------------------------------*/
  double *S_fail_obs; 
  double *S_pass_obs; 
  double dummy1, dummy2;
  S_fail_obs = &dummy1;
  S_pass_obs = &dummy2;
  *S_fail_obs = 0.0;
  *S_pass_obs = 0.0; 
  Nbins = histogram -> GetNbinsX();
  
  //-----------Perform Quality Checks on Input-------------
  if(!histogram)  {result = 5; return 0.0;}//exit if histo does not exist
  if(epsilon_min <= 0.0 || epsilon_min >= 1.0) 
                  {result = 5; return 0.0;}//exit if epsilon_min not in (0,1)
  if(epsilon_max <= 0.0 || epsilon_max >= 1.0) 
                  {result = 5; return 0.0;}//exit if epsilon_max not in (0,1)
  if(epsilon_max < epsilon_min) 
                  {result = 5; return 0.0;}//exit if max < min
  if(S_fail < 0 ) {result = 5; return 0.0;}//exit if Significance < 0
  if( S_pass < 0) {result = 5; return 0.0;}//exit if Significance < 0
  int Nentries = (int) histogram -> GetEntries();
  if(Nentries < 1){result = 4; return 0.0;}//exit if histo has 0 entries

  //-----------Find best value for occupancy b----------------
  double b = 0.0;
  int NusedBins = 0;
  for(int i = 1; i <= Nbins; i++)          //loop over all bins
    {
      if(ExclusionMask[i-1] != 1)          //do not check if bin excluded (=1)
	{
	  b += histogram -> GetBinContent(i);
	  NusedBins += 1;                  //keep track of # checked bins
	}
    }
  b *= 1.0 / (double) NusedBins;           //average for poisson stats

  //-----------Calculate Statistical Significance-------------
  double S_pass_obs_min = 0.0, S_fail_obs_max = 0.0; 
  // allocate Nbins of memory for FailedBins 
  for(int i = 0; i <= 1; i++ ) FailedBins[i] = new double [Nbins]; 
  // remember to delete[] FailedBins[0] and delete[] FailedBins[1]
  for(int i = 1; i <= Nbins; i++ )         //loop (again) over all bins
    { 
      FailedBins[0][i-1] = 0.0;            //initialise obs fraction
      FailedBins[1][i-1] = 0.0;            //initialise obs significance
      if(ExclusionMask[i-1] != 1)          //do not check if bin excluded (=1)
	{
	  //determine significance for bin to fail or pass, given occupancy
	  //hypothesis b with tolerance epsilon_min < b < epsilon_max
	  PoissionLogLikelihoodRatio(histogram->GetBinContent(i),
				     b, 
				     epsilon_min, epsilon_max,
				     S_fail_obs, S_pass_obs);
	  //set S_fail_obs to maximum over all non-excluded bins
	  //set S_pass_obs to non-zero minimum over all non-excluded bins
	  if(S_fail_obs_max == 0.0 && *S_pass_obs > 0.0)
	    S_pass_obs_min = *S_pass_obs;  //init to first non-zero value
	  if(*S_fail_obs > S_fail_obs_max) S_fail_obs_max = *S_fail_obs;
	  if(*S_pass_obs < S_pass_obs_min) S_pass_obs_min = *S_pass_obs;
	  //set FailedBins[0][] to fraction away from fitted line b
	  //set to zero if bin is within tolerance (via initialisation)
	  if(*S_fail_obs > 0) FailedBins[0][i-1] = 
				histogram->GetBinContent(i)/b - 1.0;
	  //set FailedBins[1][] to observed significance of failure
	  //set to zero if bin is within tolerance (via initialisation)
	  if(*S_fail_obs > 0) FailedBins[1][i-1] = *S_fail_obs;
	}
    }
  *S_fail_obs = S_fail_obs_max;
  *S_pass_obs = S_pass_obs_min;
  if( *S_fail_obs > 0.0 )
    {                           
      if( *S_fail_obs > S_fail )     
        {result = 1; return 0.0;}           //exit if statistically fails rule
      else
	{result = 3; return 0.0;}           //exit if non-stat significant result
    }
  else                             
    {                              
      if( *S_pass_obs > S_pass ) 
        {result = 0; return 1.0;}           //exit if statistically passes rule
      else
	{result = 2; return 0.0;}           //exit if non-stat significant result
    }
}
//---------------------------------------------------------------------------
