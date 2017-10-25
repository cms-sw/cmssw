#ifndef HistogramGenerator_H
#define HistogramGenerator_H
#include "FastSimulation/Utilities/interface/BaseNumericalRandomGenerator.h"

#include "TH1.h"

/** Numerical Random Generator for Gamma distribution.
 *  Copy of LandauFluctuations
 */

class RandomEngine;
class TAxis;

class HistogramGenerator : public BaseNumericalRandomGenerator
{
 public:

  /// Constructor : initialization of the Random Generator
   HistogramGenerator(TH1 * histo) :
     BaseNumericalRandomGenerator(histo->GetXaxis()->GetXmin(),
				  histo->GetXaxis()->GetXmax(),
				  100000,
				  3),
     myHisto(histo),
     theXaxis(histo->GetXaxis()),
     nbins(histo->GetXaxis()->GetNbins())
  {

    //    std::cout << "Old xmin/xmax = " << xmin << " " << xmax << std::endl;
    // Drop lowest and highest empty bins 
    double du = (xmax-xmin)/(float)nbins;
    // Restrict xmin to meaningful values
    while ( function(xmin) <= 0. ) xmin += du;
    // Restrict xmax to meaningful values
    while ( function(xmax) <= 0. ) xmax -= du;

    if ( xmin != histo->GetXaxis()->GetXmin() ) xmin -= du;
    if ( xmax != histo->GetXaxis()->GetXmax() ) xmax += du;

    //    std::cout << "New xmin/xmax = " << xmin << " " << xmax << std::endl;

    //    std::cout <<" Init " << std::endl;
    initialize();
  }

  /// Default destructor
  ~HistogramGenerator() override {}

  /// The probability density function implementation
  double function(double x) override { return ersatzt(x); }

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
