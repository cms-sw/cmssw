#include "DQMServices/Core/src/QStatisticalTests.h"
#include <TMath.h>

using namespace TMath;

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
      *S_fail_obs = sqrt( 2.0 * LogQ );
      *S_pass_obs = 0.0;
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
