#include "DataFormats/BTauReco/interface/MinMeanMax.h"

double MinMeanMax::min() const
{
  return min_;
}

double MinMeanMax::mean() const
{
  return mean_;
}

double MinMeanMax::max() const
{
  return max_;
}

bool MinMeanMax::isValid() const
{
  return valid_;
}

MinMeanMax::MinMeanMax () : min_(0.), mean_(0.), max_(0.), valid_(false) {}

MinMeanMax::MinMeanMax ( double min, double mean, double max ) :
  min_(min), mean_(mean), max_(max), valid_(true) {}
