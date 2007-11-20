#ifndef Muon_HLTMuonPointingFilter_h
#define Muon_HLTMuonPointingFilter_h

/** \class HLTMuonPointingFilter
 *
 * HLTFilter to select muons that points to a cylinder of configurable radius
 * and lenght.
 *
 * $Date: 07/11/2007 15:14:23 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

/* Collaborating Class Declarations */
class Propagator;

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

    mutable Propagator* thePropagator;

};
#endif // Muon_HLTMuonPointingFilter_h

