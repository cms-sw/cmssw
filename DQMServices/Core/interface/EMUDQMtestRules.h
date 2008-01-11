#include <TH2F.h>
#include <TH1F.h>
#include <TAxis.h>
#include <TMath.h>
#include <iostream.h>

//---------------------------------------------------------------------------
void BinLogLikelihoodRatio(long Nentries, long Nfailures, double epsilon_max, 
                        double* S_fail_obs, double* S_pass_obs )
{
/*--------------------------------------------------------------------------+
 |      Description:  Log-likelihood Ratio for Binomial PDF                 |
 +--------------------------------------------------------------------------+
 |                 Input to this function                                   |
 +--------------------------------------------------------------------------+
 |int Nentries          : The number of attempts                            |
 |int Nfailures         : The number of failures                            |
 |double epsilon_max,   : maximum allowed failure rate fraction             |
 |double* S_fail_obs    : uninitialised Significance of failure             |
 |double* S_pass_obs    : uninitialised Significance of Success             | 
 +--------------------------------------------------------------------------+
 |                 Return values for this function                          |
 +--------------------------------------------------------------------------+
 |double* S_fail_obs    : the observed Significance of failure              |
 |double* S_pass_obs    : the observed Significance of Success              | 
 +--------------------------------------------------------------------------+
 | Author: Richard Cavanaugh, University of Florida                         |
 | email:  Richard.Cavanaugh@cern.ch                                        |
 | Creation Date: 11.July.2005                                              |
 | Last Modified: 17.Jan.2006                                                 |
 | Comments:                                                                |
 +--------------------------------------------------------------------------*/
  long N = Nentries, n = Nfailures;
  if( n == 0 ) n = 1;   //protect against no failures: approx. 0 by 1
  if( n == N ) n -= 1;  //protect against all failures: approx. n by (n - 1)
  double epsilon_meas = (double) n / (double) N;

  double LogQ = 
    ((double)      n ) * ( Log(epsilon_meas)       - Log(epsilon_max)       ) +
    ((double) (N - n)) * ( Log(1.0 - epsilon_meas) - Log(1.0 - epsilon_max) );

  //x-check:  var of binomial = epsilon_max * (1 - epsilon_max) / N 
  if( Nentries <= 1 )   //guard against insufficient entries
    {
      *S_fail_obs = 0.0;
      *S_pass_obs = 0.0;
    }
  else if( Nfailures == 0 && ( epsilon_max <= 1.0 / (double) Nentries ) )
    {
      *S_fail_obs = 0.0;
      *S_pass_obs = 0.0;
    }
  else if( Nfailures == 0 )
    {
      *S_fail_obs = 0.0;
      *S_pass_obs = sqrt( 2.0 * LogQ );
    }
  else if( Nfailures == Nentries )
    {
      *S_fail_obs = sqrt( 2.0 * LogQ );
      *S_pass_obs = 0.0;
    }
  else if( epsilon_meas >= epsilon_max )
    {
      *S_fail_obs = 0.0;
      *S_pass_obs = sqrt( 2.0 * LogQ );
    }
  else 
    {
      *S_fail_obs = 0.0;
      *S_pass_obs = sqrt( 2.0 * LogQ );
    }
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void PoissionLogLikelihoodRatio(double data, double hypothesis,
			double epsilon_max, double epsilon_min, 
			double* S_fail_obs, double* S_pass_obs )
{ 
/*--------------------------------------------------------------------------+
 |      Description:  Log-likelihood Ratio for Poission PDF                 |
 +--------------------------------------------------------------------------+
 |                 Input to this function                                   |
 +--------------------------------------------------------------------------+
 |double data,          : The observed number of entries                    |
 |double sigma,         : The uncertainty on, data, the observed entries    |
 |double hypothesis,    : The assumed hypothese, tested against data        | 
 |double epsilon_max,   : Maximum tolerance above fraction of fitted line   |
 |double epsilon_min,   : Minimum tolerance below fraction of fitted line   |
 |double* S_fail_obs    : uninitialised Significance of failure             |
 |double* S_pass_obs    : uninitialised Significance of Success             | 
 +--------------------------------------------------------------------------+
 |                 Return values for this function                          |
 +--------------------------------------------------------------------------+
 |double* S_fail_obs    : the observed Significance of failure              |
 |double* S_pass_obs    : the observed Significance of Success              | 
 +--------------------------------------------------------------------------+
 | Author: Richard Cavanaugh, University of Florida                         |
 | email:  Richard.Cavanaugh@cern.ch                                        |
 | Creation Date: 14.Jan.2006                                              |
 | Last Modified: 16.Jan.2006                                               |
 | Comments:                                                                |
 +--------------------------------------------------------------------------*/
  double tolerance_min = hypothesis*(1.0-epsilon_min); 
  double tolerance_max = hypothesis*(1.0+epsilon_max);
  *S_pass_obs = 0.0;
  *S_fail_obs = 0.0;
  if( data > tolerance_max )
    {
      double Nsig = data - tolerance_max;
      double Nbak = tolerance_max;
      double LogQ = (double) (Nsig + Nbak) * 
	Log( 1.0 + (double) Nsig / (double) Nbak ) - (double) Nsig;
      *S_fail_obs = sqrt( 2.0 * LogQ );
    }
  else if( tolerance_min < data && data < tolerance_max ) 
    {
      if( data - hypothesis > 0.0 ) 
	{
	  double Nsig = tolerance_max - data;
	  double Nbak = tolerance_max;
	  double LogQ = (double) (Nsig + Nbak) * 
	    Log( 1.0 + (double) Nsig / (double) Nbak ) - (double) Nsig;
	  *S_pass_obs = sqrt( 2.0 * LogQ );
	}
      else 
	{
	  double Nsig =  data - tolerance_min;
	  double Nbak = tolerance_min;
	  double LogQ = (double) (Nsig + Nbak) * 
	    Log( 1.0 + (double) Nsig / (double) Nbak ) - (double) Nsig;
	  *S_pass_obs = sqrt( 2.0 * LogQ );
	}
    }
  else // data < tolerance_min 
    {
      double Nsig = tolerance_min - data;
      double Nbak = tolerance_min;
      double LogQ = (double) (Nsig + Nbak) * 
	Log( 1.0 + (double) Nsig / (double) Nbak ) - (double) Nsig;
      *S_fail_obs = sqrt( 2.0 * LogQ );
    }
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
int RuleAllContentWithinFixedRange( TH1F* histogram, double x_min, 
				    double x_max, double epsilon_max, 
				    double S_fail, double S_pass, 
				    double* epsilon_obs, 
                                    double* S_fail_obs, 
                                    double* S_pass_obs )
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
 |                 Return values for this function                          |
 +--------------------------------------------------------------------------+
 |int                   : return code                                       |
 |                        "0" = "Passed Rule & is statistically significant"|
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
  *epsilon_obs = 0.0;
  *S_fail_obs = 0.0;
  *S_pass_obs = 0.0; 
  
  //-----------Perform Quality Checks on Input-------------
  if( !histogram  )           return 5;   //exit if histo does not exist
  TAxis* xAxis = histogram -> GetXaxis(); //retrieve x-axis information
  if( x_min < xAxis -> GetXmin() || xAxis -> GetXmax() < x_max ) 
                              return 5;   //exit if x range not in hist range
  if( epsilon_max <= 0.0 || epsilon_max >= 1.0 ) 
                              return 5;   //exit if epsilon_max not in (0,1)
  if( S_fail < 0 )            return 5;   //exit if Significance < 0
  if( S_pass < 0 )            return 5;   //exit if Significance < 0

  *S_fail_obs = 0.0; *S_pass_obs = 0.0;//initialise Sig return values
  int Nentries = (int) histogram -> GetEntries();
  if( Nentries < 1 )          return 4;    //exit if histo has 0 entries

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
  *epsilon_obs        = (double)Nfailures / (double)Nentries; 

  //-----------Calculate Statistical Significance-------------
  BinLogLikelihoodRatio(Nentries,Nfailures,epsilon_max,S_fail_obs,S_pass_obs );
  if( Nfailures > Nepsilon_max )
    {                           
      if( *S_fail_obs > S_fail )     
        return 1;                         //exit if statistically fails rule
      else
	return 3;                         //exit if non-stat significant result
    }
  else                             
    {                              
      if( *S_pass_obs > S_pass ) 
        return 0;                         //exit if statistically passes rule
      else
	return 2;                         //exit if non-stat significant result
    }
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
int RuleAllContentWithinFloatingRange( TH1F* histogram, int Nrange,
                                    double epsilon_max, 
				    double S_fail, double S_pass, 
				    double* epsilon_obs, 
                                    double* S_fail_obs, 
                                    double* S_pass_obs )
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
 |                 Return values for this function                          |
 +--------------------------------------------------------------------------+
 |int                   : return code                                       |
 |                        "0" = "Passed Rule & is statistically significant"|
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
  *epsilon_obs = 0.0;
  *S_fail_obs = 0.0;
  *S_pass_obs = 0.0; 
  
  //-----------Perform Quality Checks on Input-------------
  if( !histogram  )           return 5;   //exit if histo does not exist
  int Nbins = histogram -> GetNbinsX();
  if( Nrange > Nbins )        return 5;   //exit if Nrange > # bins in histo

  if( epsilon_max <= 0.0 || epsilon_max >= 1.0 ) 
                              return 5;   //exit if epsilon_max not in (0,1)
  if( S_fail < 0 )            return 5;   //exit if Significance < 0
  if( S_pass < 0 )            return 5;   //exit if Significance < 0

  *S_fail_obs = 0.0; *S_pass_obs = 0.0;//initialise Sig return values
  int Nentries = (int) histogram -> GetEntries();
  if( Nentries < 1 )          return 4;    //exit if histo has 0 entries

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
  *epsilon_obs        = (double)Nfailures / (double)Nentries; 

  //-----------Calculate Statistical Significance-------------
  BinLogLikelihoodRatio(Nentries,Nfailures,epsilon_max,S_fail_obs,S_pass_obs );
  if( Nfailures > Nepsilon_max )
    {                           
      if( *S_fail_obs > S_fail )     
        return 1;                         //exit if statistically fails rule
      else
	return 3;                         //exit if non-stat significant result
    }
  else                             
    {                              
      if( *S_pass_obs > S_pass ) 
        return 0;                         //exit if statistically passes rule
      else
	return 2;                         //exit if non-stat significant result
    }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
int RuleAllContentAlongDiagonal( TH2F* histogram, double epsilon_max, 
				    double S_fail, double S_pass, 
				    double* epsilon_obs, 
                                    double* S_fail_obs, 
                                    double* S_pass_obs )
{
/*
 +--------------------------------------------------------------------------+
 |                 Input to this function                                   |
 +--------------------------------------------------------------------------+
 |TH2* histogram,       : histogram to be compared with Rule                |
 |double epsilon_max,   : maximum allowed failure rate fraction             |
 |double S_fail,        : required Significance to fail rule                |
 |double S_pass,        : required Significance to pass rule                |
 |double* epsilon_obs   : uninitialised actual failure rate fraction        |
 |double* S_fail_obs    : uninitialised Statistical Significance of failure |
 |double* S_pass_obs    : uninitialised Significance of Success             | 
 +--------------------------------------------------------------------------+
 |                 Return values for this function                          |
 +--------------------------------------------------------------------------+
 |int                   : return code                                       |
 |                        "0" = "Passed Rule & is statistically significant"|
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
 | Creation Date: 11.July.2005                                              |
 | Last Modified: 16.Jan.2006                                               |
 | Comments:                                                                |
 +--------------------------------------------------------------------------+
*/
  //-----------Perform Quality Checks on Input-------------
  if( !histogram  )           return 5;   //exit if histo does not exist
  if( histogram -> GetNbinsX() != histogram -> GetNbinsY() ) 
                              return 5;   //exit if histogram not square
  if( epsilon_max <= 0.0 || epsilon_max >= 1.0 ) 
                              return 5;   //exit if epsilon_max not in (0,1)
  if( S_fail < 0 )            return 5;   //exit if Significance < 0
  if( S_pass < 0 )            return 5;   //exit if Significance < 0

  *S_fail_obs = 0.0; *S_pass_obs = 0.0;//initialise Sig return values
  int Nentries = (int) histogram -> GetEntries();
  if( Nentries < 1 )          return 4;   //exit if histo has 0 entries
  //-----------Find number of successes and failures-------------
  int Nsuccesses = 0;
  for( int i = 0; i <= histogram -> GetNbinsX() + 1; i++)//count the number of
      {                                   //entries contained along diag.
        Nsuccesses += (int) histogram -> GetBinContent(i,i);
      }
  int Nfailures       = Nentries - Nsuccesses;
  double Nepsilon_max = (double)Nentries * epsilon_max;
  *epsilon_obs        = (double)Nfailures / (double)Nentries; 
  //-----------Calculate Statistical Significance-------------
  BinLogLikelihoodRatio(Nentries,Nfailures,epsilon_max,S_fail_obs,S_pass_obs );
  if( Nfailures > Nepsilon_max )
    {                           
      if( *S_fail_obs > S_fail )     
        return 1;                         //exit if statistically fails rule
      else
	return 3;                         //exit if non-stat significant result
    }
  else                             
    {                              
      if( *S_pass_obs > S_pass ) 
        return 0;                         //exit if statistically passes rule
      else
	return 2;                         //exit if non-stat significant result
    }
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
int RuleFlatOccupancy1d( TH1F* histogram, int* ExclusionMask,
			 double epsilon_min, double epsilon_max, 
			 double S_fail, double S_pass, 
			 double FailedBins[][2] )
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
 |double[][2] FailedBins: uninit. vector of bins out of tolerance           | 
 +--------------------------------------------------------------------------+
 |                 Return values for this function                          |
 +--------------------------------------------------------------------------+
 |int                   : return code                                       |
 |                        "0" = "Passed Rule & is statistically significant"|
 |                        "1" = "Failed Rule & is statistically significant"|
 |                        "2" = "Passed Rule & not stat. significant"       |
 |                        "3" = "Failed Rule & not stat. significant"       |
 |                        "4" = "zero histo entries, can not evaluate Rule" |
 |                        "5" = "Input invalid,      can not evaluate Rule" |
 |double[][2] FailedBins: the obs. vector of bins out of tolerance          | 
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
  int Nbins = histogram -> GetNbinsX();
  
  //-----------Perform Quality Checks on Input-------------
  if( !histogram  )           return 5;   //exit if histo does not exist
  if( epsilon_min <= 0.0 || epsilon_min >= 1.0 ) 
                              return 5;   //exit if epsilon_min not in (0,1)
  if( epsilon_max <= 0.0 || epsilon_max >= 1.0 ) 
                              return 5;   //exit if epsilon_max not in (0,1)
  if( epsilon_max < epsilon_min ) 
                              return 5;   //exit if max < min
  if( S_fail < 0 )            return 5;   //exit if Significance < 0
  if( S_pass < 0 )            return 5;   //exit if Significance < 0

  int Nentries = (int) histogram -> GetEntries();
  if( Nentries < 1 )          return 4;    //exit if histo has 0 entries

  //-----------Find number of successes and failures-------------
  double b = 0.0;
  int NusedBins = 0;
  for(int i = 1; i <= Nbins; i++)          //loop over all bins
    {
      if(ExclusionMask[i-1] != 1)          //do not check if bin excluded
	{
	  b += histogram -> GetBinContent(i);
	  NusedBins += 1;                  //keep track of # checked bins
	}
    }
  b *= 1.0 / (double) NusedBins;           //average for poisson stats

  //-----------Calculate Statistical Significance-------------
  double S_pass_obs_min = 0.0, S_fail_obs_max = 0.0; 
  for(int i = 1; i <= Nbins; i++ )         //loop (again) over all bins
    { 
      FailedBins[i-1][0] = 0.0;            //initialise obs fraction
      FailedBins[i-1][1] = 0.0;            //initialise obs significance
      if(ExclusionMask[i-1] != 1)          //do not check if bin excluded
	{
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
	  //set FailedBins[][0] to fraction away from fitted line b
	  //set to zero if bin is within tolerance
	  if(*S_fail_obs > 0) FailedBins[i-1][0] = 
				histogram->GetBinContent(i)/b - 1.0;
	  //set FailedBins[][1] to observed significance of failure
	  //set to zero if bin is within tolerance
	  if(*S_fail_obs > 0) FailedBins[i-1][1] = *S_fail_obs;
	}
    }
  *S_fail_obs = S_fail_obs_max;
  *S_pass_obs = S_pass_obs_min;
  if( *S_fail_obs > 0.0 )
    {                           
      if( *S_fail_obs > S_fail )     
        return 1;                         //exit if statistically fails rule
      else
	return 3;                         //exit if non-stat significant result
    }
  else                             
    {                              
      if( *S_pass_obs > S_pass ) 
        return 0;                         //exit if statistically passes rule
      else
	return 2;                         //exit if non-stat significant result
    }
}
//---------------------------------------------------------------------------
