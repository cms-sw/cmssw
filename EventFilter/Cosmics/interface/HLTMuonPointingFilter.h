#ifndef Muon_HLTMuonPointingFilter_h
#define Muon_HLTMuonPointingFilter_h

/** \class HLTMuonPointingFilter
 *
 * HLTFilter to select muons that points to a cylinder of configurable radius
 * and lenght.
 *
 * $Date: 2007/11/12 16:21:13 $
 * $Revision: 1.1 $
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
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:
    std::string theSTAMuonLabel; // label of muons 
    std::string thePropagatorName; // name of propagator to be used
    double theRadius;  // radius of cylinder
    double theMaxZ;    // half lenght of cylinder

    Cylinder::CylinderPointer theCyl;
    Plane::PlanePointer thePosPlane,theNegPlane;

    mutable Propagator* thePropagator;

};
#endif // Muon_HLTMuonPointingFilter_h

