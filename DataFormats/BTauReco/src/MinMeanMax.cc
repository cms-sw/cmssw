#include "DataFormats/BTauReco/interface/MinMeanMax.h"

double reco::MinMeanMax::min() const
{
  return min_;
}

double reco::MinMeanMax::mean() const
{
  return mean_;
}

double reco::MinMeanMax::max() const
{
  return max_;
}

bool reco::MinMeanMax::isValid() const
{
  return valid_;
}

reco::MinMeanMax::MinMeanMax () : min_(0.), mean_(0.), max_(0.), valid_(false) {}

reco::MinMeanMax::MinMeanMax ( double min, double mean, double max ) :
  min_(min), mean_(mean), max_(max), valid_(true) {}
