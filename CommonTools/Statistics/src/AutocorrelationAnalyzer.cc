#include "CommonTools/Statistics/interface/AutocorrelationAnalyzer.h"
#include <iostream>
#include <cassert>

AutocorrelationAnalyzer::AutocorrelationAnalyzer(int size)
: theSize(size),
  theNTotal(0),
  theMeans(size, 0),
  theCovariances(theSize, 0),
  theCorrelations(theSize, 0),
  calculated_(false)
{
}


double AutocorrelationAnalyzer::mean(int i)
{
  if(!calculated_) calculate();
  assert(i < theSize);
  return theMeans[i];
}
  

double AutocorrelationAnalyzer::covariance(int i, int j)
{
  if(!calculated_) calculate();
  assert(i<=theSize && j<=theSize);
  return theCovariances(i+1,j+1);
}


double AutocorrelationAnalyzer::correlation(int i, int j)
{
  if(!calculated_) calculate();
  assert(i<=theSize && j<=theSize);
  return theCorrelations(i+1,j+1);
}



void AutocorrelationAnalyzer::calculate()
{
  for(int k = 0; k < theSize; ++k)
  {
    theMeans[k] /= theNTotal;
    for (int kk = k; kk < theSize; kk++) 
    {
      theCovariances[k][kk] /= theNTotal;
    }
  }

  for (int k = 0; k < theSize; k++) 
  {
    for (int kk = k; kk < theSize; kk++)
    {
      theCorrelations[k][kk] = theCovariances[k][kk]
        / sqrt (theCovariances[k][k]*theCovariances[kk][kk]);
    }
  }

  calculated_ = true;
}


std::ostream & operator<<(std::ostream & os, AutocorrelationAnalyzer & aa)
{
  aa.calculate();
  os << "Means: " << std::endl << aa.theMeans << std::endl;
  os << "Covariances: " << std::endl << aa.theCovariances << std::endl;
  os << "Correlations: " << std::endl << aa.theCorrelations << std::endl;
  return os;
}

