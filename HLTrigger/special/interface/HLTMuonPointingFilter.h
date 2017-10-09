#ifndef Muon_HLTMuonPointingFilter_h
#define Muon_HLTMuonPointingFilter_h

/** \class HLTMuonPointingFilter
 *
 * EDFilter to select muons that points to a cylinder of configurable radius
 * and lenght.
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

/* Collaborating Class Declarations */
class Propagator;
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

/* C++ Headers */
#include <string>

/* ====================================================================== */

/* Class HLTMuonPointingFilter Interface */

class HLTMuonPointingFilter : public edm::EDFilter {

public:

  /// Constructor
  HLTMuonPointingFilter(const edm::ParameterSet&) ;

  /// Destructor
  ~HLTMuonPointingFilter() override ;

  /* Operations */
  bool filter(edm::Event &, edm::EventSetup const &) override;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  const edm::EDGetTokenT<reco::TrackCollection> theSTAMuonToken;
  const std::string thePropagatorName;      // name of propagator to be used
  const double theRadius;                   // radius of cylinder
  const double theMaxZ;                     // half length of cylinder

  Cylinder::CylinderPointer theCyl;
  Plane::PlanePointer thePosPlane, theNegPlane;

  mutable Propagator* thePropagator;
  unsigned long long  m_cacheRecordId;

};
#endif // Muon_HLTMuonPointingFilter_h

