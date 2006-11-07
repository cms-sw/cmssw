#include "../interface/ErrorCorrelation.h"

ErrorCorrelation::ErrorCorrelation( const pss& entry1, const pss& entry2, const ALIdouble corr ): theEntry1(entry1), theEntry2(entry2), theCorr(corr) 
{
}

void ErrorCorrelation::update( const ALIdouble corr )
{
  theCorr = corr;
}
