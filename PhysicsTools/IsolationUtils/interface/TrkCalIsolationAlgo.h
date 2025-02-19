#ifndef IsolationUtils_TrkCalIsolationAlgo_h
#define IsolationUtils_TrkCalIsolationAlgo_h
/* \class TrkCalIsolationAlgo<T1, C2>
 *
 * \author Christian Autermann, U Hamburg
 */
#include "DataFormats/Math/interface/deltaR.h"

template <typename T1, typename C2>
class TrkCalIsolationAlgo {
public:
  typedef double value_type;
  TrkCalIsolationAlgo( );
  TrkCalIsolationAlgo( double dRMin, double dRMax) : dRMin_( dRMin ), dRMax_( dRMax ) { }
  ~TrkCalIsolationAlgo() { } 
  double operator()(const T1 &, const C2 &) const;

private:
  double dRMin_, dRMax_;
};

//This source (track) already has defined outer eta and phi. 
//This is the track's end point in the tracker, this should be close
//the tracks entry into the calorimeter.
//A specialized template operator () for tracks in the CalIsolationAlgo class is not
//feasable, since the () operator cannot be overloaded.
template <typename T1, typename C2> double TrkCalIsolationAlgo<T1,C2>::
operator()(const T1 & cand, const C2 & elements) const {
  double etSum = 0;
  for( typename C2::const_iterator elem = elements.begin(); 
       elem != elements.end(); ++elem ) {
    double dR = deltaR( elem->eta(), elem->phi(), 
                        cand.outerEta(), cand.outerPhi() );
    if ( dR < dRMax_ && dR > dRMin_ ) {
      etSum += elem->et();
    }
  }
  return etSum;
}

#endif
