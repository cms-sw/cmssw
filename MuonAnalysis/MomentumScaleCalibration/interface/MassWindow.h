#ifndef MassWindow_h
#define MassWindow_h

#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"
#include <vector>

/**
 * Holds the information relative to a mass window: <br>
 * - central mass value <br>
 * - lower and upper bound <br>
 * - indexes of the resonances in the window <br>
 * - pointer to the background function <br>
 * It is used for both resonance and background windows. It stores the number of events
 * in the window that are counted with the "count" method. It is a double because the events
 * can be weighted.
 */

class MassWindow
{
public:
  MassWindow(const double & centralMass, const double & lowerBound, const double & upperBound,
             const std::vector<unsigned int> & indexes, backgroundFunctionBase * backgroundFunction) :
    centralMass_(centralMass), lowerBound_(lowerBound), upperBound_(upperBound), weightedEvents_(0.),
    indexes_(indexes), backgroundFunction_(backgroundFunction)
  {}
  // Used to count the number of events in the window
  void count(const double & mass, const double & weight = 1.)
  {
    if( mass > lowerBound_ && mass < upperBound_ ) {
      weightedEvents_ += weight;
    }
  }
  inline void resetCounter() { weightedEvents_ = 0; }
  inline bool isIn(const double & mass) { return( mass > lowerBound_ && mass < upperBound_ ); }
  inline double mass() const {return centralMass_;}
  inline double lowerBound() const {return lowerBound_;}
  inline double upperBound() const {return upperBound_;}
  inline double events() const {return weightedEvents_;}
  inline backgroundFunctionBase * backgroundFunction() const {return backgroundFunction_;}
  inline const std::vector<unsigned int> * indexes() const {return &indexes_;}
protected:
  double centralMass_;
  double lowerBound_;
  double upperBound_;
  // Number of events in the window
  double weightedEvents_;
  // Indexes of the resonances in this window
  std::vector<unsigned int> indexes_;
  backgroundFunctionBase * backgroundFunction_;
};

#endif
