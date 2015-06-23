#ifndef SimpleHistogramGenerator_H
#define SimpleHistogramGenerator_H

/**
 * This class is a simpler version (less memory demanding but slightly 
 * slower) of HistogramGenerator. provides a C++ version of the REMT package from 
 * Ronald Kleiss. 
 *
 * The method generate() generates random number according to the 
 * histogram bin content. 
 *
 * \author Patrick Janot
 * $Date: 16 July 2009 14:20 */

#include <vector>

class RandomEngineAndDistribution;
class TH1;
class TAxis;

class SimpleHistogramGenerator
{
 public:

  /// Constructor that perform the necessary integration and inversion steps
  /// xmin and xmax are the generation bounds, n is the internal table size
  /// and iter is the number of iterations for the numerical part.
  SimpleHistogramGenerator(TH1 * histo);

  /// Default destructor
  virtual ~SimpleHistogramGenerator() {}

  /// The random generation
  double generate(RandomEngineAndDistribution const*) const;

  int binarySearch(const int& n, 
		   const std::vector<float>& array, 
		   const double& value) const;

 private:

  /// Pointer to the histogram
  //TH1 * myHisto;

   /// the axis
  //TAxis * theXaxis;

  /// Number of bins
  int nBins;
 
  // Limits of integration
  double xMin, xMax;

  // Bin width
  double binWidth;

  /// Integral
  std::vector<float> integral;

  /// Number of entries
  double nEntries;

};
#endif
