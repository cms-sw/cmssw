#include "DQMServices/Core/interface/QStatisticalTests.h"
#include "DQMServices/Core/interface/RuleAllContentWithinFixedRange.h"
#include <iostream>

using namespace std;

//---------------------------------------------------------------------------
float RuleAllContentWithinFixedRange::runTest( const TH1F* const histogram ) 
{
/*--------------------------------------------------------------------------+
 |                 Input to this function                                   |
 +--------------------------------------------------------------------------+
 |TH1F* histogram,      : histogram to be compared with Rule                |
 |double x_min,         : x range (low). Note low edge <= bin < high edge   |
 |double x_max,         : x range (high). Note low edge <= bin < high edge  |
 |double epsilon_max,   : maximum allowed failure rate fraction             |
 |double S_fail,        : required Statistical Significance to fail rule    |
 |double S_pass,        : required Significance to pass rule                |
 |double* epsilon_obs   : uninitialised observed failure rate fraction      |
 |double* S_fail_obs    : uninitialised Significance of failure             |
 |double* S_pass_obs    : uninitialised Significance of Success             | 
 +--------------------------------------------------------------------------+
 |                 Result values for this function                          |
 +--------------------------------------------------------------------------+
 |int result            : "0" = "Passed Rule & is statistically significant"|
 |                        "1" = "Failed Rule & is statistically significant"|
 |                        "2" = "Passed Rule & not stat. significant"       |
 |                        "3" = "Failed Rule & not stat. significant"       |
 |                        "4" = "zero histo entries, can not evaluate Rule" |
 |                        "5" = "Input invalid,      can not evaluate Rule" |
 |double* epsilon_obs   : the observed failure rate frac. from the histogram|
 |double* S_fail_obs    : the observed Significance of failure              |
 |double* S_pass_obs    : the observed Significance of Success              | 
 +--------------------------------------------------------------------------+
 | Author: Richard Cavanaugh, University of Florida                         |
 | email:  Richard.Cavanaugh@cern.ch                                        |
 | Creation Date: 08.July.2005                                              |
 | Last Modified: 16.Jan.2006                                               |
 | Comments:                                                                |
 |   11.July.2005 - moved the part which calculates the statistical         |
 |                  significance of the result into a separate function     |
 +--------------------------------------------------------------------------*/
  epsilon_obs = 0.0;
  S_fail_obs = 0.0;
  S_pass_obs = 0.0; 
  
  //-----------Perform Quality Checks on Input-------------
  if(!histogram)  {result = 5; return 0.0;}//exit if histo does not exist
  TAxis* xAxis = histogram -> GetXaxis();   //retrieve x-axis information
  if(x_min < xAxis -> GetXmin() || xAxis -> GetXmax() < x_max) 
                  {result = 5; return 0.0;}//exit if x range not in hist range
  if(epsilon_max <= 0.0 || epsilon_max >= 1.0) 
                  {result = 5; return 0.0;}//exit if epsilon_max not in (0,1)
  if(S_fail < 0)  {result = 5; return 0.0;}//exit if Significance < 0
  if(S_pass < 0)  {result = 5; return 0.0;}//exit if Significance < 0
  S_fail_obs = 0.0; S_pass_obs = 0.0;       //initialise Sig return values
  int Nentries = (int) histogram -> GetEntries();
  if(Nentries < 1){result = 4; return 0.0;}//exit if histo has 0 entries

  //-----------Find number of successes and failures-------------
  int low_bin, high_bin;                   //convert x range to bin range
  if( x_min != x_max)                      //Note: x in [low_bin, high_bin)
    {                                      //Or:   x in [0,high_bin) && 
                                           //           [low_bin, max_bin]
      low_bin  = (int)( histogram -> GetNbinsX() / 
                 (xAxis -> GetXmax() - xAxis -> GetXmin()) * 
                 (x_min - xAxis -> GetXmin()) ) + 1;
      high_bin = (int)( histogram -> GetNbinsX() / 
                 (xAxis -> GetXmax() - xAxis -> GetXmin()) * 
                 (x_max - xAxis -> GetXmin()) ) + 1;
    }
  else                                     //convert x point to particular bin
    {
      low_bin = high_bin = (int)( histogram -> GetNbinsX() / 
                 (xAxis -> GetXmax() - xAxis -> GetXmin()) * 
                 (x_min - xAxis -> GetXmin()) ) + 1;
    }
  int Nsuccesses = 0;
  if(low_bin <= high_bin)                  //count number of entries
    for( int i = low_bin; i <= high_bin; i++) //in bin range
      Nsuccesses += (int) histogram -> GetBinContent(i);
  else                                     //include wrap-around case
    {
      for( int i = 0; i <= high_bin; i++)  
	Nsuccesses += (int) histogram -> GetBinContent(i);
      for( int i = low_bin; i <= histogram -> GetNbinsX(); i++)
	Nsuccesses += (int) histogram -> GetBinContent(i);	
    }
  int Nfailures       = Nentries - Nsuccesses;
  double Nepsilon_max = (double)Nentries * epsilon_max;
  epsilon_obs         = (double)Nfailures / (double)Nentries; 

  //-----------Calculate Statistical Significance-------------
  BinLogLikelihoodRatio(Nentries,Nfailures,epsilon_max,&S_fail_obs,&S_pass_obs );
  if( Nfailures > Nepsilon_max )
    {                           
      if( S_fail_obs > S_fail )     
        {result = 1; return 0.0;}           //exit if statistically fails rule
      else
	{result = 3; return 0.0;}           //exit if non-stat significant result
    }
  else                             
    {                              
      if( S_pass_obs > S_pass ) 
        {result = 0; return 1.0;}           //exit if statistically passes rule
      else
	{result = 2; return 0.0;}           //exit if non-stat significant result
    }
}
//---------------------------------------------------------------------------
