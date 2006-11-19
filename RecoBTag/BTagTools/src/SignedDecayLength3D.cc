#include <string>

#include "RecoBTag/BTagTools/interface/SignedDecayLength3D.h"

#include "Geometry/Surface/interface/Line.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Matrix/Vector.h"

using namespace std;
using namespace reco;

pair<bool,Measurement1D> SignedDecayLength3D::apply(const TransientTrack & transientTrack,
                 const GlobalVector & direction, const  Vertex & vertex)  {

  double theError=0.;
  bool theIsValid;

  //TrajectoryStateOnSurface TSOS = (aRecTrack).impactPointStateOnSurface();
  TrajectoryStateOnSurface TSOS = transientTrack.impactPointState();
  FreeTrajectoryState * FTS = TSOS.freeTrajectoryState();

  TrajectoryStateOnSurface theTSOS = closestApproachToJet(*FTS, vertex, direction,transientTrack.field());
  theIsValid= theTSOS.isValid();

  if(theIsValid){
    GlobalVector J = direction.unit();
    GlobalPoint vertexPosition(vertex.x(),vertex.y(),vertex.z());
  
    double theValue = J.dot(theTSOS.globalPosition()-vertexPosition);

    //error calculation

    HepVector j(3);
    j[0] = J.x();
    j[1] = J.y();
    j[2] = J.z();
    HepVector jj(6);
    jj[0] = J.x();
    jj[1] = J.y();
    jj[2] = J.z();
    jj[3] =0.;
    jj[4] =0.;
    jj[5] =0.;///chech it!!!!!!!!!!!!!!!!!!!!!!!
    double E1 = (theTSOS.cartesianError().matrix()).similarity(jj);
  //  double E2 = (aJet.vertex().positionError().matrix()).similarity(j);
    double E2 = RecoVertex::convertError(vertex.covariance()).matrix().similarity(j);

    theError = sqrt(E1+E2);
    

//cout<< "Error ="<< theError<<endl;
    Measurement1D A(theValue, theError);
    return pair<bool,Measurement1D>(theIsValid,A);
  }else{
    return pair<bool,Measurement1D>(theIsValid,Measurement1D(0.,0.));
  }// endif (isValid)
}// end constructor declaration


TrajectoryStateOnSurface SignedDecayLength3D::closestApproachToJet(const FreeTrajectoryState & aFTS,const Vertex & vertex, const GlobalVector& aJetDirection,const MagneticField * field) {

  GlobalVector J =aJetDirection.unit();

  Line::PositionType pos(GlobalPoint(vertex.x(),vertex.y(),vertex.z()));
  Line::DirectionType dir(J);
  Line Jet(pos,dir);

  AnalyticalTrajectoryExtrapolatorToLine TETL(field);

  return TETL.extrapolate(aFTS, Jet);
}
