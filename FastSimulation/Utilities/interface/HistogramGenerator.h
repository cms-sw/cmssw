#ifndef HistogramGenerator_H
#define HistogramGenerator_H
#include "FastSimulation/Utilities/interface/BaseNumericalRandomGenerator.h"

#include "TH1.h"
#include "TAxis.h"
#include <iostream>
#include <cmath>
/** Numerical Random Generator for Gamma distribution.
 *  Copy of LandauFluctuations
 */

class HistogramGenerator : public BaseNumericalRandomGenerator
{
 public:

  /// Constructor : initialization of the Random Generator
   HistogramGenerator(TH1 * histo) : BaseNumericalRandomGenerator(histo->GetXaxis()->GetXmin(),histo->GetXaxis()->GetXmax()),myHisto(histo),theXaxis(histo->GetXaxis()),nbins(histo->GetXaxis()->GetNbins())
  {
    //    std::cout <<" Init " << std::endl;
    initialize();
  }

  /// Default destructor
  virtual ~HistogramGenerator() {}

  /// The probability density function implementation
  virtual double function(double x) { return ersatzt(x); }

 private:
  /// Pointer to the histogram
  TH1 * myHisto;

   /// the axis
  TAxis * theXaxis;

  /// n bins
  int nbins;

  /// Gamma Function
   double ersatzt(double x);

   
};

#endif
