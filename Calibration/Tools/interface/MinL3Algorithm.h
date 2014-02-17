#ifndef MinL3Algorithm_H
#define MinL3Algorithm_H

/** \class MinL3Algorithm
 *  Implementation of the L3 Collaboration algorithm to solve a system Ax = B
 *  by minimization of |Ax-B| using an iterative linear approach
 *  This class is specific for the ECAL calibration
 *
 * 13.03.2007: R.Ofierzynski
 *  - implemented event weighting
 *
 * $Date: 2010/08/06 20:24:06 $
 * $Revision: 1.5 $
 * \author R.Ofierzynski, CERN
 */

#include <vector>
#include <iostream>


class MinL3Algorithm
{
public:
  /// Default constructor
  /// kweight_ = event weight, squareMode_ = side length of the cluster square
  MinL3Algorithm(float kweight_ = 0., int squareMode_ = 5, int mineta_ = 1, int maxeta_ = 85, int minphi_ = 1, int maxphi_ = 20);

  /// Destructor
  ~MinL3Algorithm();

  /// method doing the full calibration running nIter number of times, recalibrating the event matrix after each iteration with the new solution
  /// returns the vector of calibration coefficients built from all iteration solutions
  /// >> to be used also as recipe on how to use the calibration methods one-by-one <<
  std::vector<float> iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<int>& VmaxCeta, const std::vector<int>& VmaxCphi, const std::vector<float>& energyVector, const int& nIter, const bool& normalizeFlag = false);


  /// add event to the calculation of the calibration vector
  void addEvent(const std::vector<float>& eventSquare, const int& maxCeta, const int& maxCphi, const float& energy);

  /// recalibrate before next iteration: give previous solution vector as argument
  std::vector<float> recalibrateEvent(const std::vector<float>& eventSquare, const int& maxCeta, const int& maxCphi, const std::vector<float>& recalibrateVector); 

  /// get the solution at the end of the calibration
  std::vector<float> getSolution(bool resetsolution=true);

  /// reset for new iteration
  void resetSolution(); 

  /// method to translate from square indices to region indices
  int indexSqr2Reg(const int& sqrIndex, const int& maxCeta, const int& maxCphi);


private:

  float kweight;
  int squareMode;
  int mineta, maxeta, minphi, maxphi;
  int countEvents;
  int Nchannels, Nxtals;
  std::vector<float> wsum;
  std::vector<float> Ewsum;

};

#endif // MinL3Algorithm_H
