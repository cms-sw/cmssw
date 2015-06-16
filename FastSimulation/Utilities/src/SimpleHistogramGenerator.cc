#include "FastSimulation/Utilities/interface/SimpleHistogramGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

#include <cmath>
#include "TH1.h"
// #include <iostream>

SimpleHistogramGenerator::SimpleHistogramGenerator(TH1 * histo) :
  //myHisto(histo),
  //theXaxis(histo->GetXaxis()),
  nBins(histo->GetXaxis()->GetNbins()),
  xMin(histo->GetXaxis()->GetXmin()),
  xMax(histo->GetXaxis()->GetXmax()),
  binWidth((xMax-xMin)/(float)nBins)
{
  integral.reserve(nBins+2);
  integral.push_back(0.);
  for ( int i=1; i<=nBins; ++i )
    integral.push_back(integral[i-1]+histo->GetBinContent(i));
  integral.push_back(integral[nBins]);
  nEntries = integral[nBins+1];
  for ( int i=1; i<=nBins; ++i )
    integral[i] /= nEntries;

}


double 
SimpleHistogramGenerator::generate(RandomEngineAndDistribution const* random) const {

  // return a random number distributed according the histogram bin contents.
  // NB Only valid for 1-d histograms, with fixed bin width.

   double r1 = random->flatShoot();
   int ibin = binarySearch(nBins,integral,r1);
   double x = xMin + (double)(ibin) * binWidth;
   if (r1 > integral[ibin]) x +=
      binWidth*(r1-integral[ibin])/(integral[ibin+1] - integral[ibin]);
   return x;

}

int 
SimpleHistogramGenerator::binarySearch(const int& n, 
				       const std::vector<float>& array, 
				       const double& value) const
{
   // Binary search in an array of n values to locate value.
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   int nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == array[middle-1]) return middle-1;
      if (value  < array[middle-1]) nabove = middle;
      else                          nbelow = middle;
   }
   return nbelow-1;
}
