#ifndef GenericMinL3Algorithm_H
#define GenericMinL3Algorithm_H

/** \class GenericMinL3Algorithm
 *  Implementation of the L3 Collaboration algorithm to solve a system Ax = B
 *  by minimization of |Ax-B| using an iterative linear approach
 *
 * $Date: 2010/08/06 20:24:06 $
 * $Revision: 1.2 $
 * \author R.Ofierzynski, CERN
 */

#include <vector>
#include <iostream>


class GenericMinL3Algorithm
{
public:
  /// Default constructor
  /// CAVEAT: use normalise = true only if you know what you're doing!
  GenericMinL3Algorithm(bool normalise = false);

  /// Destructor
  ~GenericMinL3Algorithm();

  /// run the Minimization L3 Algorithm "nIter" number of times, recalibrating the event matrix after each iteration with the new solution
  /// returns the vector of calibration coefficients built from all iteration solutions
  std::vector<float> iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<float>& energyVector, int nIter);

  /// perform one iteration using the Minimization L3 Algorithm
  /// returns the vector of calibration coefficients
  std::vector<float> iterate(const std::vector<std::vector<float> >& eventMatrix, const std::vector<float>& energyVector);

protected:


private:
  bool normaliseFlag;

};

#endif // GenericMinL3Algorithm_H
