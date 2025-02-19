#ifndef Statistics_Accumulator_h
#define Statistics_Accumulator_h

/** For validation purposes.  This program
    calculates mean and RMS of a distribution

  \Author Rick Wilkinson
*/
#include <iosfwd>
#include<cmath>

class Accumulator
{
public:
  Accumulator();

  void addEntry(double value, double weight=1.);

  double mean() const;

  double variance() const;

  double sigma() const {return std::sqrt(variance());}

  double weightedMean() const;

  unsigned long nEntries() const {return n_;}

private:
  double sum_;
  double sumOfSquares_;
  double weightedSum_;
  double sumOfWeights_;
  unsigned long n_;
};

std::ostream & operator<<(std::ostream & os, const Accumulator & stat);

#endif

