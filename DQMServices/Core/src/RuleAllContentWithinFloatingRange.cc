#include "DQMServices/Core/interface/QStatisticalTests.h"
#include "DQMServices/Core/interface/RuleAllContentWithinFloatingRange.h"
#include <iostream>

using namespace std;

float RuleAllContentWithinFloatingRange::runTest( const TH1F* const histogram )
{
/*--------------------------------------------------------------------------+
 |                 Input to this function                                   |
 +--------------------------------------------------------------------------+
 |TH1F* histogram,      : histogram to be compared with Rule                |
 |int     Nrange,       : number of contiguous bins holding entries         |
 |double  epsilon_max,  : maximum allowed failure rate fraction             |
 |double  S_fail,       : required Statistical Significance to fail rule    |
 |double  S_pass,       : required Significance to pass rule                |
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
 | Creation Date: 07.Jan.2006                                               |
 | Last Modified: 16.Jan.2006                                               |
 | Comments:                                                                |
 +--------------------------------------------------------------------------*/
  epsilon_obs = 0.0;
  S_fail_obs = 0.0;
  S_pass_obs = 0.0; 
  
  //-----------Perform Quality Checks on Input-------------
  if(!histogram)    {result = 5; return 0.0;}//exit if histo does not exist
  int Nbins = histogram -> GetNbinsX();
  if(Nrange > Nbins){result = 5; return 0.0;}//exit if Nrange > # bins in histo
  if(epsilon_max <= 0.0 || epsilon_max >= 1.0) 
                    {result = 5; return 0.0;}//exit if epsilon_max not in (0,1)
  if(S_fail < 0)    {result = 5; return 0.0;}//exit if Significance < 0
  if(S_pass < 0)    {result = 5; return 0.0;}//exit if Significance < 0
  S_fail_obs = 0.0; S_pass_obs = 0.0;         //initialise Sig return values
  int Nentries = (int) histogram -> GetEntries();
  if(Nentries < 1)  {result = 4; return 0.0;}//exit if histo has 0 entries

  //-----------Find number of successes and failures-------------
  int Nsuccesses = 0, EntriesInCurrentRange = 0;
  for( int i = 1; i <= Nrange; i++ )  //initialise Nsuccesses 
    {                                 //histos start with bin index 1 (not 0)
      Nsuccesses += (int) histogram -> GetBinContent(i);
    }
  EntriesInCurrentRange = Nsuccesses;
  for( int i = Nrange + 1; i <= Nbins; i++ ) //optimise floating bin range
    { //slide range by adding new high side bin & subtracting old low side bin
      EntriesInCurrentRange += 
	(int) ( histogram -> GetBinContent(i) -
		histogram -> GetBinContent(i - Nrange) );
      if(EntriesInCurrentRange > Nsuccesses)
	Nsuccesses = EntriesInCurrentRange;
    }
  for( int i = 1; i < Nrange; i++ ) //include possiblity of wrap-around
    { //slide range by adding new low side bin & subtracting old high side bin
      EntriesInCurrentRange += 
	(int) ( histogram -> GetBinContent(i) -
		histogram -> GetBinContent(Nbins - (Nrange - i) ) );
      if(EntriesInCurrentRange > Nsuccesses)
	Nsuccesses = EntriesInCurrentRange;
    }
  int Nfailures       = Nentries - Nsuccesses;
  double Nepsilon_max = (double)Nentries * epsilon_max;
  epsilon_obs        = (double)Nfailures / (double)Nentries; 

  //-----------Calculate Statistical Significance-------------
  BinLogLikelihoodRatio(Nentries,Nfailures,epsilon_max,&S_fail_obs,&S_pass_obs );
  if( Nfailures > Nepsilon_max )
    {                           
      if( S_fail_obs > S_fail )     
        {result = 1; return 0.0;}        //exit if statistically fails rule
      else
	{result = 3; return 0.0;}        //exit if non-stat significant result
    }
  else                             
    {                              
      if( S_pass_obs > S_pass ) 
        {result = 0; return 1.0;}        //exit if statistically passes rule
      else
	{result = 2; return 0.0;}        //exit if non-stat significant result
    }
}
//---------------------------------------------------------------------------
