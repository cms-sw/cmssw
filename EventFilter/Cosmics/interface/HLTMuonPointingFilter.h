#ifndef Muon_HLTMuonPointingFilter_h
#define Muon_HLTMuonPointingFilter_h

/** \class HLTMuonPointingFilter
 *
 * HLTFilter to select muons that points to a cylinder of configurable radius
 * and lenght.
 *
 * $Date: 2012/01/21 14:56:54 $
 * $Revision: 1.4 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

/* Collaborating Class Declarations */
class Propagator;
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

/* C++ Headers */
#include <string>

/* ====================================================================== */

/* Class HLTMuonPointingFilter Interface */

class HLTMuonPointingFilter : public HLTFilter {

public:
  
  /// Constructor
  HLTMuonPointingFilter(const edm::ParameterSet&) ;
  
  /// Destructor
  ~HLTMuonPointingFilter() ;
  
  /* Operations */ 
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
  
private:
  std::string theSTAMuonLabel; // label of muons 
  std::string thePropagatorName; // name of propagator to be used
    double theRadius;  // radius of cylinder
  double theMaxZ;    // half lenght of cylinder
  
  Cylinder::CylinderPointer theCyl;
  Plane::PlanePointer thePosPlane,theNegPlane;
  
  mutable Propagator* thePropagator;
  unsigned long long  m_cacheRecordId;
  
};
#endif // Muon_HLTMuonPointingFilter_h

