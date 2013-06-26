#ifndef GenericHouseholder_h
#define GenericHouseholder_h

/** \class GenericHouseholder
 *  Generic implementation of the QR decomposition of a matrix using Householder transformation
 *
 * $Date: 2010/08/06 20:24:06 $
 * $Revision: 1.2 $
 * \author Lorenzo Agostino, R.Ofierzynski, CERN
 */

#include <vector>
#include <iostream>

class GenericHouseholder
{
public:
  /// Default constructor
  /// CAVEAT: use normalise = true only if you know what you're doing!
  GenericHouseholder(bool normalise = false);

  /// Destructor
  ~GenericHouseholder();

  /// run the Householder Algorithm several times (nIter). Returns the final vector of calibration coefficients.
  std::vector<float> iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<float>& energyVector, const int nIter);

  /// run the Householder Algorithm. Returns the vector of calibration coefficients.
  std::vector<float> iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<float>& energyVector);

private:
  /// make decomposition
  /// input: m=number of events, n=number of channels, qr=event matrix
  /// output: qr = new event matrix, alpha, pivot
  /// returns a boolean value, true if decomposition worked, false if it didn't
  bool decompose(const int m, const int n, std::vector<std::vector<float> >& qr,  std::vector<float>& alpha, std::vector<int>& pivot);  

  /// Apply transformations to rhs
  /// output: r = ?, y = solution
  void solve(int m, int n, const std::vector<std::vector<float> > &qr, const std::vector<float> &alpha, const std::vector<int> &pivot, std::vector<float> &r, std::vector<float> &y);

  bool normaliseFlag;
};

#endif
