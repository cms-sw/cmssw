/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/
//This class is taken from the GEANT4 tool kit  and changed!!!!!

#include "GeneratorInterface/Hydjet2Interface/interface/RandArrayFunction.h"

RandArrayFunction::RandArrayFunction(const double *aProbFunc, int theProbSize, int intType)
  : fNBins(theProbSize), 
    fInterpolationType(intType)
{
  PrepareTable(aProbFunc);
}

RandArrayFunction::RandArrayFunction(int theProbSize, int intType)
  : fNBins(theProbSize), 
    fInterpolationType(intType)
{}

void RandArrayFunction::PrepareTable(const double* aProbFunc) {
  //Prepares fIntegralPdf.
  if(fNBins < 1) {
    edm::LogError("RandArrayFunction") << "RandArrayFunction constructed with no bins - will use flat distribution";
    UseFlatDistribution();
    return;
  }

  fIntegralPdf.resize(fNBins + 1);
  fIntegralPdf[0] = 0;
  int ptn;
  for (ptn = 0; ptn < fNBins; ++ptn ) {
    double weight = aProbFunc[ptn];
    if (weight < 0.) {
      // We can't stomach negative bin contents, they invalidate the 
      // search algorithm when the distribution is fired.
      edm::LogWarning("RandArrayFunction") << "RandArrayFunction constructed with negative-weight bin "<< ptn << " == " << weight << " -- will substitute 0 weight";
      weight = 0.;
    }
    fIntegralPdf[ptn + 1] = fIntegralPdf[ptn] + weight;
  }

  if (fIntegralPdf[fNBins] <= 0.) {
    edm::LogWarning("RandArrayFunction") << "RandArrayFunction constructed with nothing in bins - will use flat distribution";
    UseFlatDistribution();
    return;
  }

  for (ptn = 0; ptn < fNBins + 1; ++ptn)
    fIntegralPdf[ptn] /= fIntegralPdf[fNBins];
  
  // And another useful variable is ...
  fOneOverNbins = 1.0 / fNBins;
  // One last chore:
  if (fInterpolationType && fInterpolationType != 1) {
    edm::LogInfo("RandArrayFunction") << "RandArrayFunction does not recognize fInterpolationType "<< fInterpolationType << " Will use type 0 (continuous linear interpolation)";
    fInterpolationType = 0;
  }
} 

void RandArrayFunction::UseFlatDistribution() {
  //Called only by PrepareTable in case of user error. 
  fNBins = 1;
  fIntegralPdf.resize(2);
  fIntegralPdf[0] = 0;
  fIntegralPdf[1] = 1;
  fOneOverNbins = 1.0;
} 

double RandArrayFunction::MapRandom(double rand) const {
  // Private method to take the random (however it is created) and map it
  // according to the distribution.

  int nBelow = 0;	  // largest k such that I[k] is known to be <= rand
  int nAbove = fNBins;  // largest k such that I[k] is known to be >  rand
  int middle;
     
  while (nAbove > nBelow+1) {
    middle = (nAbove + nBelow+1)>>1;
    rand >= fIntegralPdf[middle] ? nBelow = middle : nAbove = middle;
  }// after this loop, nAbove is always nBelow+1 and they straddle rad:
       
  /*assert ( nAbove = nBelow+1 );
    assert ( fIntegralPdf[nBelow] <= rand );
    assert ( fIntegralPdf[nAbove] >= rand );*/  
  // If a defective engine produces rand=1, that will 
  // still give sensible results so we relax the > rand assertion

  if (fInterpolationType == 1) {
    return nBelow * fOneOverNbins;
  } 
  else {
    double binMeasure = fIntegralPdf[nAbove] - fIntegralPdf[nBelow];
    // binMeasure is always aProbFunc[nBelow], 
    // but we don't have aProbFunc any more so we subtract.
    
    if (!binMeasure) { 
      // rand lies right in a bin of measure 0.  Simply return the center
      // of the range of that bin.  (Any value between k/N and (k+1)/N is 
      // equally good, in this rare case.)
      return (nBelow + .5) * fOneOverNbins;
    }

    double binFraction = (rand - fIntegralPdf[nBelow]) / binMeasure;
    
    return (nBelow + binFraction) * fOneOverNbins;
  }
} 

void RandArrayFunction::FireArray(int size, double *vect) const {
  for (int i = 0; i < size; ++i)
    vect[i] = Fire();
}
