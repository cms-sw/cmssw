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
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

/* Collaborating Class Declarations */
class Propagator;
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

/* C++ Headers */
#include <string>
#include <memory>

/* ====================================================================== */

/* Class HLTMuonPointingFilter Interface */

class HLTMuonPointingFilter : public edm::global::EDFilter<> {
public:
  /// Constructor
  HLTMuonPointingFilter(const edm::ParameterSet &);

  /// Destructor
  ~HLTMuonPointingFilter() override;

  /* Operations */
  bool filter(edm::StreamID, edm::Event &, edm::EventSetup const &) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const edm::EDGetTokenT<reco::TrackCollection> theSTAMuonToken;

  const std::string
      thePropagatorName;  // name of propagator to be used  const edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorToken;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorToken;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theMGFieldToken;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theTrackingGeometryToken;

  const double theRadius;          // radius of cylinder
  const double theMaxZ;            // half length of cylinder
  const unsigned int thePixHits;   // number of pixel hits
  const unsigned int theTkLayers;  // number of tracker layers with measurements
  const unsigned int theMuonHits;  // number of valid muon hits

  const Cylinder::CylinderPointer theCyl;
  const Plane::PlanePointer thePosPlane;
  const Plane::PlanePointer theNegPlane;
};
#endif  // Muon_HLTMuonPointingFilter_h
