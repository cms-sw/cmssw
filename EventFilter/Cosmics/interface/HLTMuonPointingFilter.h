#ifndef Muon_HLTMuonPointingFilter_h
#define Muon_HLTMuonPointingFilter_h

/** \class HLTMuonPointingFilter
 *
 * HLTFilter to select muons that points to a cylinder of configurable radius
 * and lenght.
 *
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
  const std::string theSTAMuonLabel;        // label of muons
  const std::string thePropagatorName;      // name of propagator to be used
  const double theRadius;                   // radius of cylinder
  const double theMaxZ;                     // half length of cylinder

  Cylinder::CylinderPointer theCyl;
  Plane::PlanePointer thePosPlane, theNegPlane;

};
#endif // Muon_HLTMuonPointingFilter_h

