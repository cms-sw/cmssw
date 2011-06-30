#include "TH1.h"

TH1F *th1fmorph(const char *chname, 
                const char *chtitle,
                TH1F *hist1,TH1F *hist2,
                Double_t par1,Double_t par2,Double_t parinterp,
                Double_t morphedhistnorm,
                Int_t idebug=0) ;
TH1D *th1fmorph(const char *chname, 
                const char *chtitle,
                TH1D *hist1,TH1D *hist2,
                Double_t par1,Double_t par2,Double_t parinterp,
                Double_t morphedhistnorm,
                Int_t idebug=0) ;
  //--------------------------------------------------------------------------
  // Author           : Alex Read 
  // Version 0.2 of ROOT implementation, 08.05.2011
  // *
  // *      Perform a linear interpolation between two histograms as a function
  // *      of the characteristic parameter of the distribution.
  // *
  // *      The algorithm is described in Read, A. L., "Linear Interpolation
  // *      of Histograms", NIM A 425 (1999) 357-360.
  // *      
  // *      This ROOT-based CINT implementation is based on the FORTRAN77
  // *      implementation used by the DELPHI experiment at LEP (d_pvmorph.f).
  // *      The use of double precision allows pdf's to be accurately 
  // *      interpolated down to something like 10**-15.
  // *
  // *      The input histograms don't have to be identical, the binning is also
  // *      interpolated.
  // *
  // *      Extrapolation is allowed (a warning is given) but the extrapolation 
  // *      is not as well-defined as the interpolation and the results should 
  // *      be used with great care.
  // *
  // *      Data in the underflow and overflow bins are completely ignored. 
  // *      They are neither interpolated nor do they contribute to the 
  // *      normalization of the histograms.
  // *
  // * Input arguments:
  // * ================
  // * chname, chtitle : The ROOT name and title of the interpolated histogram.
  // *                   Defaults for the name and title are "THF1-interpolated"
  // *                   and "Interpolated histogram", respectively.
  // *
  // * hist1, hist2    : The two input histograms.
  // *
  // * par1,par2       : The values of the linear parameter that characterises
  // *                   the histograms (e.g. a particle mass).
  // *
  // * parinterp       : The value of the linear parameter we wish to 
  // *                   interpolate to. 
  // * 
  // * morphedhistnorm : The normalization of the interpolated histogram 
  // *                   (default is 1.0).  
  // * 
  // * idebug          : Default is zero, no internal information displayed. 
  // *                   Values between 1 and increase the verbosity of 
  // *                   informational output which may prove helpful to
  // *                   understand errors and pathalogical results.
  // * 
  // * The routine returns a pointer (TH1 *) to a new histogram which is
  // * the interpolated result.
  // *
  // *------------------------------------------------------------------------
  // Changes from 0.1 to 0.2:
  // o Treatment of empty and non-existing histograms now well-defined.
  // o The tricky regions of the first and last bins are improved (and
  //   well-tested).
  // *------------------------------------------------------------------------

