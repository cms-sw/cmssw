//----------Author's Name: B.Fabbro DSM/DAPNIA/SPP CEA-Saclay
//----------Copyright: Those valid for CEA sofware
//----------Modified: 07/06/2007

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TDistrib.h"

#include <math.h>
#include "Riostream.h"

//------------------------- TDistrib.cxx ----------------------
//
//   Creation: 01 Jul  2002
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------

ClassImp(TDistrib)
//_____________________________________________________________________________
//
// TDistrib: 1D distribution for a random variable. This class contains
//           methods to built the distribution of the variable (1D array
//           of N values, N = number of events) and to calculate quantities
//           of interest (expectation values, variances, correlation with
//           another variable, etc...)
// 

TDistrib::TDistrib()
{
//constructor without argument: empty distribution
  fCnew    = 0;
  fCdelete = 0;

  fNbEvents = 0;
  fValues   = 0;    // Null pointer. No allocation

  //........................ error tags
  fCodeErrMax = 10;

  fMemoErrEv    = 0;
  fMemoErrVar   = 0;
  fMemoErrSdev  = 0;
  fMemoErrCovar = 0;
}

void  TDistrib::fCopy(const TDistrib& dis)
{
//private copy

  fNbEvents = dis.fNbEvents;
  
  for(Int_t i = 0 ; i < fNbEvents ; i++)
    {
      fValues[i] = dis.fValues[i];
    }  
}

#define CONCOP
#ifndef CONCOP

TDistrib::TDistrib(const TDistrib& dis)
{
//copy constructor

  if ( dis.fNbEvents > 0 ) 
    {
      fValues = new Double_t[dis.fNbEvents];  fCnew++;     // allocation
      
      // cout << "*Constructor TDistrib::TDistrib(const TDistrib& dis)*> "
      //      << endl << " allocation of " << fNbEvents
      //      << " 'Double_t' for array member at memory adress:"
      //      << " &fValues = " << &fValues << endl;
      
      fCopy(dis);     // call to the private copy
    }
  else
    { 
      cerr << endl
	   << "*TDistrib::TDistrib(const TDistrib& dis) *** ERROR ***>" << endl
	   << " ------------------------------------------------------"
	   << endl
	   << " You try to fill a Distribution with"
	   << " NULL or NEGATIVE number of events:" << endl
	   << " fNbEvents = " << fNbEvents 
	   << endl << endl;
      
      fValues = 0;    // Null pointer. No allocation
      
      cout << "*Constructor TDistrib::TDistrib()*> no memory"
	   << " allocation for array member" << endl;
    }
}

#endif // CONCOP

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    overloading of the operator=
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TDistrib& TDistrib::operator=(const TDistrib& dis)
{
//overloading of the operator=

  if ( fValues != dis.fValues )
    {
      if (fNbEvents != dis.fNbEvents )
	{
	  if ( fValues != 0 ) {delete [] fValues;  fCdelete++;}
	  fNbEvents = dis.fNbEvents;
	  fValues = new Double_t[fNbEvents];   fCnew++;     // allocation
	  
	  // cout << "*Overloading TDistrib::operator=(const TDistrib& dis)*> "
	  //      << endl << " allocation of " << fNbEvents
	  //      << " 'Double_t' for array member at memory adress:"
	  //      << " &fValues = " << &fValues << endl;	  
	}
      fCopy(dis);        // call to the private copy
    }
  return *this;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    constructors with arguments
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//               Genuine initialization of a distribution from
//               a given positive number of events (nbv) and
//               from an array of values (val[] or *val).
//

//---------------- Constructor with the distribution in a 1D array of Double_t

TDistrib::TDistrib(const Int_t&  nbv , const  Double_t *val)
{
//constructors with arguments
//Genuine initialization of a distribution from
//a given positive number of events (nbv) and
//from an array of values.
// THE DIMENSION OF val MUST BE GREATER THAN OR EQUAL TO nbv 
//

  Init();

  fNbEvents = nbv;

  if ( fNbEvents > 0 ) 
    {
      fValues =  new Double_t[fNbEvents];         fCnew++;
      
      //  cout << "*Constructor TDistrib::TDistrib(const Int_t&  nbv"
      //       << " , const  Double_t *val)*> " << endl
      //       << " allocation of " << fNbEvents
      //       << " 'Double_t' for array member at memory adress:"
      //       << " &fValues = " << &fValues << endl;
      
      for(Int_t i = 0 ; i < fNbEvents ; i++)
	{
	  fValues[i] = val[i];
	}
    }
  else
    {
      cerr << endl
	   << "*Constructor TDistrib::TDistrib(const Int_t&  nbv"
	   << " , const  Double_t *val) *** ERROR ***> " << endl
	   << " ------------------------------------------------"
	   << "--------------------------------------" << endl
	   << " You try to fill a distribution with"
	   << " NULL or NEGATIVE number of events:" << endl
	   << " fNbEvents = " << fNbEvents 
	   << endl << endl;
      
      fValues = 0;    // Null pointer. No allocation
      
      cout << "*Constructor TDistrib::TDistrib(const Int_t&  nbv"
	   << " , const  Double_t *val)> " << endl
	   << "  no memory allocation for array member" << endl;
    }
}

//---------------------- constructor with the distribution in a TVectorD 

TDistrib::TDistrib(const Int_t&  nbv , const  TVectorD val)
{
//constructors with arguments
//Genuine initialization of a distribution from
//a given positive number of events (nbv) and
//from an array of values.
// THE DIMENSION OF val MUST BE GREATER THAN OR EQUAL TO nbv 

  Init();

  fNbEvents = nbv;

  if ( fNbEvents > 0 ) 
    {
      fValues =  new Double_t[fNbEvents];         fCnew++;
      
      //  cout << "*Constructor TDistrib::TDistrib(const Int_t&  nbv"
      //       << " , const  Double_t *val)*> " << endl
      //       << " allocation of " << fNbEvents
      //       << " 'Double_t' for array member at memory adress:"
      //       << " &fValues = " << &fValues << endl;
      
      for(Int_t i = 0 ; i < fNbEvents ; i++)
	{
	  fValues[i] = (Double_t)val(i);
	}
    }
  else
    {
      cerr << endl
	   << "*Constructor TDistrib::TDistrib(const Int_t&  nbv"
	   << " , const  TVectorD val) *** ERROR ***> " << endl
	   << " ------------------------------------------------"
	   << "--------------------------------------" << endl
	   << " You try to fill a distribution with"
	   << " NULL or NEGATIVE number of events:" << endl
	   << " fNbEvents = " << fNbEvents 
	   << endl << endl;
      
      fValues = 0;    // Null pointer. No allocation
      
      cout << "*Constructor TDistrib::TDistrib(const Int_t&  nbv"
	   << " , const  TVectorD val)> " << endl
	   << "  no memory allocation for array member" << endl;
    }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                            destructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TDistrib::~TDistrib()
{
//destructor

  fNbEvents = 0;
  
  if ( fValues != 0)
    {
      // cout << "*Destructor TDistrib::~TDistrib()*> " << endl
      //      << " release memory for array member allocated at adress:"
      //      << " &fValues = " << &fValues
      //      << endl;

      delete [] fValues;      fCdelete++;      // release memory
    }
  else
    {
      // cout << "*Destructor TDistrib::~TDistrib()*> " << endl
      //      << " no release memory for array member"
      //      << " since no allocation was done (fValues = " << fValues
      //      << ") " << endl;            
    } 

  if ( fCnew != fCdelete )
    {
      cout << "*TDistrib> WRONG MANAGEMENT OF ALLOCATIONS: fCnew = "
	   << fCnew << ", fCdelete = " << fCdelete << endl;
    }
  else
    {
      //  cout << "*TDistrib> BRAVO! GOOD MANAGEMENT OF ALLOCATIONS: fCnew = "
      //	   << fCnew << ", fCdelete = " << fCdelete << endl;
    }

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                           M E T H O D S
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

void TDistrib::Init()
{
//Miscellaneous initializations

  //................... Init counters new/delete
  fCnew    = 0;
  fCdelete = 0;
  //................init memo error tags

  fCodeErrMax = 10;

  fMemoErrEv    = 0;
  fMemoErrVar   = 0;
  fMemoErrSdev  = 0;
  fMemoErrCovar = 0;
}

//-------------------------------------------------------------------
//
//  Refill: change the values in the array (TVectorD) if the number
//          of events is the same as before
//
//-------------------------------------------------------------------
Bool_t  TDistrib::Refill(const Int_t& nmax, const  TVectorD val)
{
//Refill: change the values

  Bool_t ok_refill = kFALSE;

  if (nmax == fNbEvents)
    {
      for(Int_t i = 0 ; i < fNbEvents ; i++)
	{
	  fValues[i] = (Double_t)val(i);
	}
      ok_refill = kTRUE;
    }
  else
    {
      cout << "!TDistib::Refill(...)> nmax = " << nmax << ". Not equal to fNbEvents ( = "
	   << fNbEvents << "). No Refill" << endl; 
    }
  return ok_refill;
}

//--------------------------------------------------------------------------
//
//      Resize: change the number of events if lower than before
//
//      KEEP THE INITIAL ALLOCATION => CHANGE ONLY THE MAX NUMBER OF EVENT
//      So that the calcultations (means, variances, correlations, ...)
//      are done on this new number of events.
//
//      RETURN VALUE:  kTRUE  => a resize has been done
//                     kFALSE => no resize was necessary
//
//--------------------------------------------------------------------------

Bool_t  TDistrib::Resize(const Int_t& nmax)
{  
//Resize:change the number of events if lower than before

  Bool_t ok_resize = kFALSE;

  if( nmax > 0 && nmax < fNbEvents )
    {
      fNbEvents = nmax;
      ok_resize = kTRUE;
    }
  else
    {
      // if( nmax > fNbEvents ){cout << "!TDistib::Resize(...)> new nmax > old nmax. No Resize" << endl;}
      // if( nmax < 0         ){cout << "!TDistib::Resize(...)> new nmax < 0. No Resize" << endl;}
      // if( nmax == 0        ){cout << "!TDistib::Resize(...)> new nmax = 0. No Resize" << endl;}
    }
  return ok_resize;
}

//-----------------------------------------------
//
//      Return of the number of events
//
//-----------------------------------------------

Int_t  TDistrib::NumberOfEvents()
{
//Return of the number of events
  return fNbEvents;
}

//-----------------------------------------------
//
//      Return of the pointer to the values
//
//-----------------------------------------------

Double_t*  TDistrib::VariableValues()
{
//Return of the pointer to the values
  return fValues;
}

//------------------------------------------------------------
//
//     Calculation and return of the expectation value
//
//------------------------------------------------------------

Double_t TDistrib::ExpectationValue()
{
//Calculation and return of the expectation value for nmax events

  Int_t nmax = fNbEvents;
  
  Double_t vmoy = (Double_t)0.;
  
  for (Int_t n = 0 ; n < nmax ; n++)
    {
      //vmoy = vmoy + fValues[n];
      vmoy += fValues[n];
    }

  if(nmax > 0)
    {  
      vmoy = vmoy/(Double_t)nmax;
    }
  else
    {
      cout << "*TDistrib::ExpectationValue() *** ERROR ***> " << endl
	   << " Calculation of expectation value for a Distribution"
	   << endl << " with a non-positive number of events."
	   << " nmax = " << nmax << endl;
      //      {Int_t cintoto; cin >> cintoto;}
    }

  if( !(vmoy >= 0 || vmoy < 0) )
    {
      if ( fMemoErrEv < fCodeErrMax )
	{
	  cout << "*TDistrib::ExpectationValue() *** ERROR ***>"
	       << " EXTRA expectation value! vmoy = "
	       << vmoy << " *==> MAY BE SOMETHING WRONG IN MEMORY (?)..." 
	       << '\007' << endl;
	  fMemoErrEv++;
	}
      if ( fMemoErrEv == fCodeErrMax )
	{
	  cout << "*TDistrib::ExpectationValue()> This message has been issued "
	       << fMemoErrEv <<" times. It will be suppressed from now" << endl
	       << "                               BUT THE PROBLEM WHICH CAUSES"
	       << " THIS MESSAGE CAN STILL BE PRESENT." << endl;
	  fMemoErrEv++;
	}
    }
  return vmoy;
}

//------------------------------------------------------------
//
//     Calculation and return of the deviation values
//      
//------------------------------------------------------------

Double_t* TDistrib::DeviationValues()  
{
//Calculation and return of the deviation values

  Int_t nmax = fNbEvents;

  Double_t  vmoy   = ExpectationValue();

  Double_t* ecarts = new Double_t[nmax];      fCnew++;
  
  for (Int_t n = 0 ; n < nmax ; n++)
    {
      ecarts[n] = fValues[n] - vmoy;
    }
  
  TDistrib  deviat( nmax, ecarts);
  
  delete [] ecarts;                                 fCdelete++;
  
  return deviat.fValues;
}

//------------------------------------------------------------
//
//          Calculation and return of the variance
//
//------------------------------------------------------------

Double_t TDistrib::VarianceValue()  
{
//Calculation and return of the variance

  Int_t     nmax = fNbEvents;
  Double_t  vmoy = ExpectationValue();
  
  Double_t*   tab_squared_deviat = new Double_t[nmax];      fCnew++;
  
  for (Int_t n = 0 ; n < nmax ; n++)
    {
      Double_t deviat = fValues[n] - vmoy;
      tab_squared_deviat[n] = deviat*deviat;
    }
  
  TDistrib  sq_deviat( nmax, tab_squared_deviat);
   
  Double_t var = sq_deviat.ExpectationValue();
  
  if ( !(var >= 0) )
    {
      if ( var < 0 )
	{
	  cout << "*TDistrib::VarianceValue() *** ERROR ***>"
	       << " NEGATIVE result at variance calculation! var = "
	       << var << '\007' << endl;
	}
      else
	{
	  if ( fMemoErrVar < fCodeErrMax )
	    {
	      cout << "*TDistrib::VarianceValue() *** ERROR ***>"
		   << " EXTRA result at variance calculation! var = "
		   << var << "*===> MAY BE SOMETHING WRONG IN MEMORY (?)..." 
		   << '\007' << endl;
	      fMemoErrVar++;
	    }
	  if ( fMemoErrVar == fCodeErrMax )
	    {
	      cout << "*TDistrib::VarianceValue()> This message has been issued "
		   << fMemoErrVar <<" times. It will be suppressed from now" << endl
		   << "BUT THE PROBLEM WHICH CAUSES THIS MESSAGE CAN STILL BE PRESENT" << endl;
 	      fMemoErrVar++;
	    }
	}
    }

  delete [] tab_squared_deviat;                           fCdelete++;
  return var;
}

//------------------------------------------------------------
//
//     Calculation and return of the standard deviation
//
//------------------------------------------------------------

Double_t TDistrib::StandardDeviation()  
{
//Calculation and return of the standard deviation

  Double_t sigma = (Double_t)0.;
  Double_t var = VarianceValue();

  if( var >= (Double_t)0. )
    {
      sigma = (Double_t)sqrt(var);
    }
  else
    {
      if( var < (Double_t)0.)
	{
	  cout << "*TDistrib::StandardDeviation() *** ERROR ***>"
	       << " detection of NEGATIVE variance! var = " << var
	       << ", standard deviation value forced to -1"
	       << '\007' << endl;
	  sigma = -1;
	}
      else
	{
	  if ( fMemoErrSdev < fCodeErrMax )
	    {
	      cout << "*TDistrib::StandardDeviation() *** ERROR ***>"
		   << " detection of EXTRA variance! var = " << var
		   << ", standard deviation value forced to -1" << endl
		   << " *===> MAY BE SOMETHING WRONG IN MEMORY (?)..." 
		   << '\007' << endl;
	      sigma = -1;
	      fMemoErrSdev++;
	    }
	  if ( fMemoErrSdev == fCodeErrMax )
	    {
	      cout << "*TDistrib::StandardDeviation()> This message has been issued "
		   << fMemoErrSdev <<" times. It will be suppressed from now" << endl
		   << "BUT THE PROBLEM WHICH CAUSES THIS MESSAGE CAN STILL BE PRESENT" << endl; 
	      fMemoErrSdev++; 
	    }
	}
    }
  return sigma;
}
//------------------------- idem with TString in argument
Double_t TDistrib::StandardDeviation(TString calling_prog)  
{
//Calculation and return of the standard deviation
//Return the argument in message in case of error

  Double_t sigma;
  Double_t var = (*this).VarianceValue();
  
  if( var >= (Double_t)0. )
    {
      sigma = sqrt(var);
    }
  else
    {
      if ( var < (Double_t)0. )
	{
	  cout << "*TDistrib::StandardDeviation() *** ERROR ***>"
	       << " detection of NEGATIVE variance! var = " << var
	       << ", standard deviation value forced to -1" << endl
	       << " The calling program has transmited the following TString argument: "
	       << calling_prog << '\007' << endl;
	  sigma = -1;
	}
      else
	{
	  cout << "*TDistrib::StandardDeviation() *** ERROR ***>"
	       << " detection of EXTRA variance! var = " << var
	       << ", standard deviation value forced to -1" << endl
	       << " The calling program has transmited the following TString argument: "
	       << calling_prog << " *===> MAY BE SOMETHING WRONG IN MEMORY (?)..." 
	       << '\007' << endl;
	  sigma = -1;
	}
    }
  return sigma;
}

//------------------------------------------------------------
//
//     Calculation and return of the covariance 
//     with another distribution
//
//------------------------------------------------------------

Double_t  TDistrib::Covariance(TDistrib& X_j)
{
//Calculation and return of the covariance with another distribution

  Double_t  covar = (Double_t)0.;
  Int_t     n_i   = (*this).fNbEvents;
  Int_t     n_j   = X_j.NumberOfEvents();

  //..... determination of the lowest number of events

  Int_t      n_min = 0;
  if (n_i <= n_j){n_min = n_i;}
  if (n_i >  n_j){n_min = n_j;}

  //................. Calculation of the covariance
  if(n_min > 0)
    {
      Double_t *prod_ecarts = new Double_t[n_min];        fCnew++;
      
      Double_t* fValues_i = (*this).VariableValues();
      Double_t* fValues_j = X_j.VariableValues();
      
      Double_t  v_moy_i = (*this).ExpectationValue();      
      Double_t  v_moy_j = X_j.ExpectationValue();  
      
      for (Int_t n = 0 ; n < n_min ; n++)
	{
	  prod_ecarts[n] = (fValues_i[n] - v_moy_i)*(fValues_j[n] - v_moy_j);
	}
      
      TDistrib  d_cov(n_min,  prod_ecarts);
      covar = d_cov.ExpectationValue();

      delete [] prod_ecarts;                              fCdelete++;
    }
  else
    {
      if ( fMemoErrCovar < fCodeErrMax )
	{
	  cout << "*TDistrib::Covariance(TDistrib& X_j)> *** ERROR ***"
	       << " Calculation of covariance with NON POSITIVE numbers of events"
	       << " Distribution i: n_i = " << n_i << ", distribution j: n_j = " << n_j
	       << '\007' << endl;
	  fMemoErrCovar++;
	}
      
      if ( fMemoErrCovar == fCodeErrMax )
	{
	  cout << "*TDistrib::Covariance(TDistrib& X_j)> This message has been issued "
	       << fMemoErrCovar <<" times. It will be suppressed from now" << endl
	       << "                               BUT THE PROBLEM WHICH CAUSES"
	       << " THIS MESSAGE CAN STILL BE PRESENT." << endl;
	  fMemoErrCovar++;
	}
    }
  return covar;
} 

//------------------------------------------------------------
//
//     Calculation and return of the correlation 
//     with another distribution
//
//------------------------------------------------------------

//----------> still to be done... (perhaps)

//------------------------------------------------------------
//
//    Building of a histogram of the distribution
//                
//------------------------------------------------------------

void TDistrib::HistoDistrib(const Int_t& nb_bins,
			    Double_t&    xmin,        Double_t& xmax,
			    Double_t*    dist_histo,  Int_t&    range_null,
			    Int_t&       n_underflow, Int_t&    n_overflow)
{
//Histogram making of the distribution.
//A small difference can be observed between the values
//of the histogram X axis and the values of the variable
//distribution (because of the binning).
//The difference is given by:
//
// | mean(histogram) - mean(distribution) | = delta_X_for_1_bin/sqrt{12} 
//
//It is small if the number of bins is at least greater than 100

  Int_t nb_events = fNbEvents;

  if ( nb_events > 0 )
    {  
      //........... Search for xmin and xmax 

      xmin = fValues[0];
      xmax = fValues[0];

      for (Int_t i_evt = 0 ; i_evt < nb_events ; i_evt++)
	{
	  if ( xmin >= fValues[i_evt] ){xmin = fValues[i_evt];}
	  if ( xmax <= fValues[i_evt] ){xmax = fValues[i_evt];}
	}

      //.......................................... histo building
      
      Int_t i_bin;
      
      //..................... reset histo
      
      for (i_bin = 0 ; i_bin < nb_bins ; i_bin++)
	{
	  dist_histo[i_bin] = (Double_t)0.;
	}
      
      //..................... binning a la ROOT
      
      Double_t range_x = xmax - xmin;
      range_null = 0;
      
      n_underflow = 0;
      n_overflow  = 0;
      
      if ( range_x > 0 )
	{
	  for (Int_t i_evt = 0 ; i_evt < nb_events ; i_evt++)
	    {
	      // i_bin = (Int_t)( ( ( fValues[i_evt] - xmin )/range_x )*( (Double_t)(nb_bins - 1) ) );
	      i_bin = (Int_t)( ( ( fValues[i_evt] - xmin )/range_x )*(Double_t)nb_bins);
	      //i_bin = floor( ( ( fValues[i_evt] - xmin )/range_x )*(Double_t)(nb_bins-1)+(Double_t)(0.5) );

	      if( i_bin < 0 )
		{
		  i_bin = 0;
		  //n_underflow = n_underflow + 1;
		  n_underflow++;
		}

	      if( i_bin >= nb_bins )
		{
		  i_bin = nb_bins - 1;
		  //n_overflow = n_overflow + 1;
		  n_overflow++;
		}
	      
	      //dist_histo[i_bin] = dist_histo[i_bin] + (Double_t)1.;
	      dist_histo[i_bin]++; 
	    }
	}
      else
	{
	  //range_null = range_null + 1;
	  range_null++;
	}
    }
  else
    {
      cerr << "*TDistrib::HistoDistrib() *** ERROR ***> " << endl
	   << " You are trying to make an histogram for a distribution"
	   << endl << " with a non-positive number of events."
	   << " nb_events = " << nb_events << '\007' << endl;
    }
}
