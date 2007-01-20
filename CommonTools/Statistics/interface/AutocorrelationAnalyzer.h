#ifndef Statistics_AutocorrelationAnalyzer_h
#define Statistics_AutocorrelationAnalyzer_h

#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include <iosfwd>

/** This class accepts objects which support the [] operator,
 *  such as a digi or a vector,
 *  and calculates the correlation matrix between the components
 */

class AutocorrelationAnalyzer
{
public:
  explicit AutocorrelationAnalyzer(int size);

  /// indexing starts from 0
  double mean(int i);
  double covariance(int i, int j);
  double correlation(int i, int j);

  template<class T>
  void analyze(const T & t)
  {
    for (int ii = 0; ii < 10; ii++) {
      theMeans[ii] += t[ii];
      for (int ij = ii; ij < 10; ij++) {
        theCovariances[ii][ij] += t[ii] * t[ij];
      }
    }
    ++theNTotal;
  }
 
  friend std::ostream & operator<<(std::ostream & os, AutocorrelationAnalyzer & aa);

private:
  void calculate();

  int theSize;
  int theNTotal;
  HepVector theMeans;
  HepSymMatrix theCovariances;
  HepSymMatrix theCorrelations;
  bool calculated_;
};

#endif

