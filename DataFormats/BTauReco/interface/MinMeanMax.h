#ifndef BTauReco_MinMeanMax_h
#define BTauReco_MinMeanMax_h

namespace reco {
  class MinMeanMax {
  public:
    /**
     *  simple class that groups minimum, mean, and maximum value
     *  of a distribution
     */
    MinMeanMax ();
    MinMeanMax ( double min, double mean, double max );
    double min() const;
    double mean() const;
    double max() const;
    bool isValid() const;
  private:
    double min_, mean_, max_;
    bool valid_;
  };
}

#endif
