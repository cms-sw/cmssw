#ifndef HouseholderDecomposition_H
#define HouseholderDecomposition_H

/** \class HouseholderDecomposition
 *  Implementation of the QR decomposition of a matrix using Householder transformation
 *
 *  13.03.2007: R.Ofierzynski
 *   - using a reduced matrix
 *   - implements Regional Householder Algorithm
 *
 * $Date: 2010/08/06 20:24:06 $
 * $Revision: 1.4 $
 * \author Lorenzo Agostino, R.Ofierzynski, CERN
 */

#include <vector>
#include <iostream>

class HouseholderDecomposition
{
public:
  /// Default constructor
  HouseholderDecomposition(int squareMode_ = 5, int mineta_ = 1, int maxeta_ = 85, int minphi_ = 1, int maxphi_ = 20);

  /// Destructor
  ~HouseholderDecomposition();

  /// Run Regional HouseholderAlgorithm (fast version), that splits matrix into regional matrices and inverts them separately.
  /// Returns the final vector of calibration coefficients.
  /// input: eventMatrix - the skimmed event matrix, 
  ///        VmaxCeta, VmaxCphi - vectors containing eta and phi indices of the maximum containment crystal for each event, 
  ///        energyVector - the energy vector, 
  ///        nIter - number of iterations to be performed, 
  ///        regLength - default length of the region (in eta- and phi-indices), regLength=5 recommended
  /// Comment from author: if you use the same events, 2 iterations are recommended; the second iteration gives corrections of the order of 0.0001
  std::vector<float> runRegional(const std::vector<std::vector<float> >& eventMatrix, const std::vector<int>& VmaxCeta, const std::vector<int>& VmaxCphi, const std::vector<float>& energyVector, const int& nIter, const int& regLength = 5);

  /// Run the Householder Algorithm several times (nIter). Returns the final vector of calibration coefficients.
  /// Comment from author: unless you do a new selection in between the iterations you don't need to run more than once;
  ///                      a second iteration on the same events does not improve the result in this case
  std::vector<float> iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<int>& VmaxCeta, const std::vector<int>& VmaxCphi, const std::vector<float>& energyVector, const int& nIter, const bool& normalizeFlag = false);

  /// Run the Householder Algorithm. Returns the vector of calibration coefficients.
  std::vector<float> iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<int>& VmaxCeta, const std::vector<int>& VmaxCphi, const std::vector<float>& energyVectorOrig);

  /// Recalibrate before next iteration: give previous solution vector as argument
  std::vector<float> recalibrateEvent(const std::vector<float>& eventSquare, const int& maxCeta, const int& maxCphi, const std::vector<float>& recalibrateVector); 

  /// Method to translate from square indices to region indices
  int indexSqr2Reg(const int& sqrIndex, const int& maxCeta, const int& maxCphi);


private:
  /// Make decomposition
  /// input: m=number of events, n=number of channels, qr=event matrix
  /// output: qr = transformed event matrix, alpha, pivot
  /// returns a boolean value, true if decomposition worked, false if it didn't
  bool decompose();  

  /// Apply transformations to rhs
  /// output: r = residual vector (energy vector), y = solution
  void solve(std::vector<float> &y);

  /// Unzips the skimmed matrix into a full matrix
  std::vector<std::vector<float> > unzipMatrix(const std::vector<std::vector<float> >& eventMatrix, const std::vector<int>& VmaxCeta, const std::vector<int>& VmaxCphi);

  /// Determines the regions used for splitting of the full matrix and calibrating separately
  /// used by the public runRegional method
  void makeRegions(const int& regLength);

  int squareMode, countEvents;
  int mineta, maxeta, minphi, maxphi, Neta, Nphi;
  int Nchannels, Nxtals, Nevents;
  std::vector< std::vector<float> > eventMatrixOrig;
  std::vector< std::vector<float> > eventMatrixProc;
  std::vector<float> energyVectorProc;
  std::vector<float> alpha;
  std::vector<int> pivot;

  std::vector <int> regMinPhi, regMaxPhi, regMinEta, regMaxEta;
  float sigmaReplacement;
};

#endif // HouseholderDecomposition_H
