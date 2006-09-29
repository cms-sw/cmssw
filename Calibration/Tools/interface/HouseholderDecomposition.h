#ifndef HouseholderDecomposition_h
#define HouseholderDecomposition_h

/** \class HouseholderDecomposition
 *  Implementation of the QR decomposition of a matrix using Householder transformation
 *
 * $Date: 2006/08/25 $
 * $Revision: 1.2 $
 * \author Lorenzo Agostino, R.Ofierzynski, CERN
 */

#include <vector>
#include <iostream>

using namespace std;

class HouseholderDecomposition
{
public:
  /// Default constructor
  /// CAVEAT: use normalise = true only if you know what you're doing!
  HouseholderDecomposition(bool normalise = false);

  /// Destructor
  ~HouseholderDecomposition();

  /// run the Householder Algorithm several times (nIter). Returns the final vector of calibration coefficients.
  vector<float> iterate(const vector<vector<float> >& eventMatrix, const vector<float>& energyVector, const int nIter);

  /// run the Householder Algorithm. Returns the vector of calibration coefficients.
  vector<float> iterate(const vector<vector<float> >& eventMatrix, const vector<float>& energyVector);

private:
  /// make decomposition
  /// input: m=number of events, n=number of channels, qr=event matrix
  /// output: qr = new event matrix, alpha, pivot
  /// returns a boolean value, true if decomposition worked, false if it didn't
  bool decompose(const int m, const int n, vector<vector<float> >& qr,  vector<float>& alpha, vector<int>& pivot);  

  /// Apply transformations to rhs
  /// output: r = ?, y = solution
  void solve(int m, int n, const vector<vector<float> > &qr, const vector<float> &alpha, const vector<int> &pivot, vector<float> &r, vector<float> &y);

  bool normaliseFlag;
};

#endif
