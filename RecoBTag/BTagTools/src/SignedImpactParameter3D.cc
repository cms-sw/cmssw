#include "DataFormats/GeometrySurface/interface/Line.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Matrix/Vector.h"
#include <string>

using namespace std;
using namespace reco;

pair<bool, Measurement1D> SignedImpactParameter3D::apply(const TransientTrack& transientTrack,
                                                         const GlobalVector& direction,
                                                         const Vertex& vertex) const {
  double theValue = 0.;
  double theError = 0.;
  bool theIsValid = false;

  TrajectoryStateOnSurface TSOS = transientTrack.impactPointState();

  if (!TSOS.isValid()) {
    cout << "====>>>> SignedImpactParameter3D::apply : TSOS not valid = " << TSOS.isValid() << endl;
    return pair<bool, Measurement1D>(theIsValid, Measurement1D(0., 0.));
  }

  const FreeTrajectoryState* FTS = TSOS.freeTrajectoryState();

  const GlobalVector& JetDirection(direction);

  TrajectoryStateOnSurface theTSOS = closestApproachToJet(*FTS, vertex, JetDirection, transientTrack.field());
  theIsValid = theTSOS.isValid();

  if (theIsValid) {
    GlobalVector D = distance(theTSOS, vertex, JetDirection);
    GlobalVector J = JetDirection.unit();
    GlobalPoint vertexPosition(vertex.x(), vertex.y(), vertex.z());
    double theDistanceAlongJetAxis = J.dot(theTSOS.globalPosition() - vertexPosition);
    theValue = D.mag() * (theDistanceAlongJetAxis / abs(theDistanceAlongJetAxis));

    GlobalVector DD = D.unit();
    GlobalPoint T0 = theTSOS.globalPosition();
    GlobalVector T1 = theTSOS.globalMomentum();
    GlobalVector TT1 = T1.unit();
    GlobalVector Xi(T0.x() - vertex.position().x(), T0.y() - vertex.position().y(), T0.z() - vertex.position().z());

    AlgebraicVector6 deriv;
    AlgebraicVector3 deriv_v;

    deriv_v[0] = -DD.x();
    deriv_v[1] = -DD.y();
    deriv_v[2] = -DD.z();

    deriv[0] = DD.x();
    deriv[1] = DD.y();
    deriv[2] = DD.z();
    deriv[3] = -(TT1.dot(Xi) * DD.x()) / T1.mag();
    deriv[4] = -(TT1.dot(Xi) * DD.y()) / T1.mag();
    deriv[5] = -(TT1.dot(Xi) * DD.z()) / T1.mag();

    double E1 = ROOT::Math::Similarity(deriv, theTSOS.cartesianError().matrix());
    double E2 = ROOT::Math::Similarity(deriv_v, vertex.covariance());
    //    double E2 = RecoVertex::convertError(vertex.covariance()).matrix().similarity(deriv_v);
    //    double E2 = 0.; // no vertex error because of stupid use of hundreds of different types for same thing
    theError = sqrt(E1 + E2);

    Measurement1D A(theValue, theError);

    return pair<bool, Measurement1D>(theIsValid, A);
  } else {
    return pair<bool, Measurement1D>(theIsValid, Measurement1D(0., 0.));
  }  // endif (isValid)
}

TrajectoryStateOnSurface SignedImpactParameter3D::closestApproachToJet(const FreeTrajectoryState& aFTS,
                                                                       const Vertex& vertex,
                                                                       const GlobalVector& aJetDirection,
                                                                       const MagneticField* field) {
  GlobalVector J = aJetDirection.unit();

  Line::PositionType pos(GlobalPoint(vertex.x(), vertex.y(), vertex.z()));
  Line::DirectionType dir(J);
  Line Jet(pos, dir);

  AnalyticalTrajectoryExtrapolatorToLine TETL(field);

  return TETL.extrapolate(aFTS, Jet);
}

GlobalVector SignedImpactParameter3D::distance(const TrajectoryStateOnSurface& aTSOS,
                                               const Vertex& vertex,
                                               const GlobalVector& aJetDirection) {
  Line::PositionType pos2(aTSOS.globalPosition());
  Line::DirectionType dir2((aTSOS.globalMomentum()).unit());
  Line T(pos2, dir2);

  GlobalPoint X = GlobalPoint(vertex.x(), vertex.y(), vertex.z());  // aVertex.position();

  GlobalVector D = T.distance(X);

  return D;
}

pair<double, Measurement1D> SignedImpactParameter3D::distanceWithJetAxis(const TransientTrack& track,
                                                                         const GlobalVector& direction,
                                                                         const Vertex& vertex) {
  double theDistanceAlongJetAxis(0.);
  double theDistanceToJetAxis(0.);
  double theLDist_err(0.);
  TrajectoryStateOnSurface TSOS = track.impactPointState();

  if (!TSOS.isValid()) {
    cout << "====>>>> SignedImpactParameter3D::distanceWithJetAxis : TSOS not valid = " << TSOS.isValid() << endl;
    return pair<double, Measurement1D>(theDistanceAlongJetAxis, Measurement1D(theDistanceToJetAxis, theLDist_err));
  }

  const FreeTrajectoryState* FTS = TSOS.freeTrajectoryState();

  const GlobalVector& jetDirection(direction);

  //
  // Check whether the track has been used in the vertex
  //

  //FIXME
  float weight = 0.;  //vertex.trackWeight(aRecTrack);

  TrajectoryStateOnSurface stateAtOrigin = track.impactPointState();
  TrajectoryStateOnSurface aTSOS = closestApproachToJet(*FTS, vertex, jetDirection, track.field());
  bool isValid = stateAtOrigin.isValid();
  //  bool IsValid= aTSOS.isValid();

  if (isValid) {
    //get the Track line at origin
    Line::PositionType pos(stateAtOrigin.globalPosition());
    Line::DirectionType dir((stateAtOrigin.globalMomentum()).unit());
    Line track(pos, dir);
    // get the Jet  line
    // Vertex vertex(vertex);
    GlobalVector jetVector = jetDirection.unit();
    Line::PositionType pos2(GlobalPoint(vertex.x(), vertex.y(), vertex.z()));
    Line::DirectionType dir2(jetVector);
    Line jet(pos2, dir2);
    // now compute the distance between the two lines
    // If the track has been used to refit the Primary vertex then sign it positively, otherwise negative

    theDistanceToJetAxis = (jet.distance(track)).mag();
    if (weight < 1)
      theDistanceToJetAxis = -theDistanceToJetAxis;

    // ... and the flight distance along the Jet axis.
    GlobalPoint V = jet.position();
    GlobalVector Q = dir - jetVector.dot(dir) * jetVector;
    GlobalVector P = jetVector - jetVector.dot(dir) * dir;
    theDistanceAlongJetAxis = P.dot(V - pos) / Q.dot(dir);

    //
    // get the covariance matrix of the vertex and compute the error on theDistanceToJetAxis
    //

    ////AlgebraicSymMatrix vertexError = vertex.positionError().matrix();

    // build the vector of closest approach between lines

    GlobalVector H((jetVector.cross(dir).unit()));

    CLHEP::HepVector Hh(3);
    Hh[0] = H.x();
    Hh[1] = H.y();
    Hh[2] = H.z();

    //  theLDist_err = sqrt(vertexError.similarity(Hh));

    //    cout << "distance to jet axis : "<< theDistanceToJetAxis <<" and error : "<< theLDist_err<<endl;
    // Now the impact parameter ...

    /*    GlobalPoint T0 = track.position();
    GlobalVector D = (T0-V)- (T0-V).dot(dir) * dir;
    double IP = D.mag();    
    GlobalVector Dold = distance(aTSOS, aJet.vertex(), jetDirection);
    double IPold = Dold.mag();
*/
  }
  Measurement1D DTJA(theDistanceToJetAxis, theLDist_err);

  return pair<double, Measurement1D>(theDistanceAlongJetAxis, DTJA);
}
