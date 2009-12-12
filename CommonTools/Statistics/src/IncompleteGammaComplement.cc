#include "CommonTools/Statistics/src/IncompleteGammaComplement.h"
#include "CommonTools/Statistics/src/GammaContinuedFraction.h"
#include "CommonTools/Statistics/src/GammaSeries.h"
#include "CommonTools/Statistics/src/GammaLn.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <iostream>
#include <cmath>

float IncompleteGammaComplement::value(float a, float x)
{
  if( x < 0.0 || a <= 0.0 ) 
    edm::LogInfo("IncompleteGammaComplement")<< "IncompleteGammaComplement::invalid arguments";
  if( x < (a+1.0) )
    // take the complement of the series representation
    return 1.-GammaSeries(a,x)*(exp(-x + a*log(x) - GammaLn(a)));
  else
    // use the continued fraction representation
    return GammaContinuedFraction(a,x)*(exp(-x + a*log(x) - GammaLn(a)));
}


float IncompleteGammaComplement::ln(float a, float x)
{
  if( x < 0.0 || a <= 0.0 ) 
edm::LogInfo("IncompleteGammaComplement")<< "IncompleteGammaComplement::invalid arguments";
  if( x < (a+1.0) )
    // take the complement of the series representation    
    return log(1.-GammaSeries(a,x)*(exp(-x + a*log(x) - GammaLn(a))));
  else
    // use the continued fraction representation
    return log(GammaContinuedFraction(a,x)) -x + a*log(x) - GammaLn(a);
}
