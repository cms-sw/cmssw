#include "CommonTools/Statistics/interface/Accumulator.h"
#include <ostream>

Accumulator::Accumulator() 
:  sum_(0.),
   sumOfSquares_(0.),
   weightedSum_(0.),
   sumOfWeights_(0.),
   n_(0)
{
}


void Accumulator::addEntry(double value, double weight) {
  sum_ += value;
  sumOfSquares_ += (value*value);
  weightedSum_ += value*weight;
  sumOfWeights_ += weight;
  ++n_;
}


double Accumulator::mean() const {
  return sum_/n_;
}


double Accumulator::variance() const {
  double numerator = sumOfSquares_ - sum_*mean();
  unsigned long denominator = n_-1;
  return numerator/denominator;
}

 
double Accumulator::weightedMean() const  {
  return weightedSum_ / sumOfWeights_;
}


std::ostream& operator<<(std::ostream & os,const Accumulator & stat) {
  os << "entries: " << stat.nEntries();
  if(stat.nEntries() > 0) {
     os << "   Mean: " << stat.mean(); 
  }
  if(stat.nEntries() > 1) {      
		 os << "   Sigma: " << stat.sigma();
  }
  return os;
}


