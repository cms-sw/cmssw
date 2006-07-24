#ifndef MinL3Algorithm_H
#define MinL3Algorithm_H

/** \class MinL3Algorithm
 *  Implementation of the L3 Collaboration algorithm to solve a system Ax = B
 *  by minimization of |Ax-B| using an iterative linear approach
 *
 * $Date: 2006/06/26 $
 * $Revision: 1.0 $
 * \author R.Ofierzynski, CERN
 */

#include <vector>
#include <iostream>

using namespace std;


class MinL3Algorithm
{
public:
  /// Default constructor
  MinL3Algorithm(bool normalise = true);

  /// Destructor
  ~MinL3Algorithm();

  /// run the Minimization L3 Algorithm "nIter" number of times, recalibrating the event matrix after each iteration with the new solution
  /// returns the vector of calibration coefficients built from all iteration solutions
  vector<float> iterate(const vector<vector<float> >& eventMatrix, const vector<float>& energyVector, int nIter);

  /// perform one iteration using the Minimization L3 Algorithm
  /// returns the vector of calibration coefficients
  vector<float> iterate(const vector<vector<float> >& eventMatrix, const vector<float>& energyVector);

protected:


private:
  bool normaliseFlag;

};

#endif // MinL3Algorithm_H
