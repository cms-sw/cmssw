#ifndef BaseNumericalRandomGenerator_H
#define BaseNumericalRandomGenerator_H

/**
 * This abstract class provides a C++ version of the REMT package from 
 * Ronald Kleiss. 
 *
 * In the constructor, the probability density function according to 
 * which random numbers have to be generated is numerically integrated
 * and inversed. The arguments are the generation bounds, the size of 
 * the internal numerical tables, and the number of iterations for 
 * integration and inversion.
 *
 * The method generate() generates random number according to the 
 * aforementioned probability density function(). The latter must be 
 * implemented in a real class inheriting from BaseNumericalRandomGenerator.
 * (This fuction cannot be abstract, because the constructor uses it.)
 * A normal flat random generation between 0 and 1 is performed otherwise.
 *
 * \author Patrick Janot
 * $Date: 12 Jan 2004 14:40 */

#include <vector>

class RandomEngine;

class BaseNumericalRandomGenerator
{
 public:

  /// Constructor that perform the necessary integration and inversion steps
  /// xmin and xmax are the generation bounds, n is the internal table size
  /// and iter is the number of iterations for the numerical part.
  BaseNumericalRandomGenerator(const RandomEngine* engine,
			       double xmin=0., 
			       double xmax=1., 
			       int n=1000, 
			       int iter=6);

  /// Default destructor
  virtual ~BaseNumericalRandomGenerator() {}

  /// The initialization (numerical integarion, inversion)
  void initialize();

  /// The random generation according to function()
  double generate() const;

  /// The random generation according to function(), refined to generate
  /// as an exponential in each of the intervals
  double generateExp() const;

  /// The random generation according to function(), refined to generate
  /// as a linear function in each of the intervals
  double generateLin() const;

  // The probability density function, to be implemented in the real class
  virtual double function(double x)=0;

  /// To shoot in a given interval
  bool setSubInterval(double x1,double x2);

 protected:

  const RandomEngine* random;

  std::vector<double> sampling;
  std::vector<double> f;
  double xmin, xmax;
  int n, iter;
  double rmin, deltar;
  
 private:

  int m;

};
#endif
