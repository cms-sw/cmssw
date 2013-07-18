#ifndef IsolationUtils_CalIsolationAlgo_h
#define IsolationUtils_CalIsolationAlgo_h
/* \class CalIsolationAlgo<T1, C2>
 *
 * \author Christian Autermann, U Hamburg
 *
 * template class to calculate calorimeter isolation, the extrapolation
 * to the calorimeter surface is optional.
 *
 */
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/IsolationAlgos/interface/PropagateToCal.h"

template <typename T1, typename C2>
class CalIsolationAlgo {
public:
  typedef double value_type;
  CalIsolationAlgo( ) { }
  CalIsolationAlgo(double dRMin, double dRMax, bool do_propagation,
                   double radius, double minZ, double maxZ, bool theIgnoreMaterial):
     dRMin_( dRMin ), dRMax_( dRMax ), do_propagation_( do_propagation ),
     SrcAtCal(radius, minZ, maxZ, theIgnoreMaterial) { }
  ~CalIsolationAlgo();
  
  void setBfield( const MagneticField * bField ) {
       bField_ = bField;  }
  double operator()(const T1 &, const C2 &) const;

private:
  double dRMin_, dRMax_;
  bool   do_propagation_;
  const MagneticField * bField_;
  PropagateToCal SrcAtCal;
};


template <typename T1, typename C2>
CalIsolationAlgo<T1,C2>::~CalIsolationAlgo() {
}

template <typename T1, typename C2> double CalIsolationAlgo<T1,C2>::
operator()(const T1 & cand, const C2 & elements) const {
  const GlobalPoint Vertex(cand.vx(), cand.vy(), cand.vz());//@@check if this is [cm]!
  //GlobalVector Cand(cand.pt(), cand.eta(), cand.phi()); 
  GlobalVector Cand(cand.px(), cand.py(), cand.pz()); 

  ///Extrapolate charged particles from their vertex to the point of entry into the
  ///calorimeter, if this is requested in the cfg file.
  if (do_propagation_ && cand.charge()!=0) 
     SrcAtCal.propagate(Vertex, Cand, cand.charge(), bField_);

  double etSum = 0;
  for( typename C2::const_iterator elem = elements.begin(); 
       elem != elements.end(); ++elem ) {
    double dR = deltaR( elem->eta(), elem->phi(), 
                        (double)Cand.eta(), (double)Cand.phi() );
    if ( dR < dRMax_ && dR > dRMin_ ) {
      etSum += elem->et();
    }
  }
  return etSum;
}

#endif
