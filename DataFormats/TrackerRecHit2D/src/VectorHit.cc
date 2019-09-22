#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackGeomDet.h"

VectorHit::VectorHit(const VectorHit& vh):
  BaseTrackerRecHit(*vh.det(), trackerHitRTTI::vector),
  thePosition(vh.localPosition()),
  theDirection(vh.localDirection()),
  theCovMatrix(vh.parametersError()),
  theChi2(vh.chi2()),
  theDimension(vh.dimension()),
  theLowerCluster(vh.lowerClusterRef()),
  theUpperCluster(vh.upperClusterRef())
{}

VectorHit::VectorHit(const GeomDet& idet,
                     const LocalPoint& posLower,
                     const LocalVector& dir,
                     const AlgebraicSymMatrix& covMatrix,
		     const double& Chi2,
                     OmniClusterRef const& lower, OmniClusterRef const& upper) :
  BaseTrackerRecHit(idet, trackerHitRTTI::vector),
  thePosition(posLower),
  theDirection(dir),
  theCovMatrix(covMatrix),
  theChi2(Chi2),
  theDimension(4),
  theLowerCluster(lower),
  theUpperCluster(upper)
{}

VectorHit::VectorHit(const GeomDet& idet, const VectorHit2D& vh2Dzx, const VectorHit2D& vh2Dzy,
                     OmniClusterRef const& lower, OmniClusterRef const& upper) :
  BaseTrackerRecHit(idet, trackerHitRTTI::vector),
  theDimension(4),
  theLowerCluster(lower),
  theUpperCluster(upper)
{
  thePosition = LocalPoint(vh2Dzx.localPosition().x(), vh2Dzy.localPosition().x(), 0.);

  theDirection = LocalVector(vh2Dzx.localDirection().x(), vh2Dzy.localDirection().x(), 1.);
  //theDirection = LocalVector(vh2Dzx.localDirection().x(), vh2Dzy.localDirection().x(), vh2Dzy.localDirection().z());
  //theDirection = theDirection.unit();

  //building the cov matrix 4x4 starting from the 2x2
  AlgebraicSymMatrix22 covMatZX = vh2Dzx.covMatrix();
  AlgebraicSymMatrix22 covMatZY = vh2Dzy.covMatrix();

  theCovMatrix=AlgebraicSymMatrix(4);
  theCovMatrix[0][0] = covMatZX[0][0];   // var(dx/dz)
  theCovMatrix[1][1] = covMatZY[0][0];   // var(dy/dz)
  theCovMatrix[2][2] = covMatZX[1][1];   // var(x)
  theCovMatrix[3][3] = covMatZY[1][1];   // var(y)
  theCovMatrix[0][2] = covMatZX[0][1];   // cov(dx/dz,x)
  theCovMatrix[1][3] = covMatZY[0][1];   // cov(dy/dz,y)

  theChi2 = vh2Dzx.chi2() + vh2Dzy.chi2();
}

bool VectorHit::sharesInput( const TrackingRecHit* other, SharedInputType what) const 
{
  if (what==all && (geographicalId() != other->geographicalId())) return false;
 
  if (!sameDetModule(*other)) return false;

  if (trackerHitRTTI::isVector(*other) ) {
    const VectorHit* otherVh = static_cast<const VectorHit*>(other);
    return sharesClusters(*this, *otherVh, what);
  }
   
  if (what==all)  return false;

  // what about multi???
  auto const & otherClus = reinterpret_cast<const BaseTrackerRecHit *>(other)->firstClusterRef();
  return (otherClus==lowerClusterRef())  ||  (otherClus==upperClusterRef());
  
}

bool VectorHit::sharesClusters(VectorHit const & h1, VectorHit const & h2,
            SharedInputType what) const {
  bool lower =  h1.lowerClusterRef()== h2.lowerClusterRef();
  bool upper =  h1.upperClusterRef()== h2.upperClusterRef();
  
  return (what==TrackingRecHit::all) ? (lower && upper) : (upper||lower);
  
} 

void VectorHit::getKfComponents4D( KfComponentsHolder & holder ) const {
  //if (!hasPositionAndError()) throwExceptionUninitialized("getKfComponents");
  AlgebraicVector4 & pars = holder.params<4>();
  pars[0] = theDirection.x();
  pars[1] = theDirection.y();
  pars[2] = thePosition.x();
  pars[3] = thePosition.y();

  AlgebraicSymMatrix44 & errs = holder.errors<4>();
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      errs(i,j) = theCovMatrix[i][j];
    }
  }

  ProjectMatrix<double,5,4>  & pf = holder.projFunc<4>();
  pf.index[0] = 1;
  pf.index[1] = 2;
  pf.index[2] = 3;
  pf.index[3] = 4;

  holder.measuredParams<4>() = AlgebraicVector4( & holder.tsosLocalParameters().At(1), 4 );
  holder.measuredErrors<4>() = holder.tsosLocalErrors().Sub<AlgebraicSymMatrix44>( 1, 1 );

}

VectorHit::~VectorHit() {}

AlgebraicVector VectorHit::parameters() const {

    // (dx/dz,dy/dz,x,y)
    AlgebraicVector result(4);

    result[0] = theDirection.x();
    result[1] = theDirection.y();
    result[2] = thePosition.x();
    result[3] = thePosition.y();
    return result;

}

Global3DPoint VectorHit::lowerGlobalPos() const {
  const StackGeomDet* stackDet = dynamic_cast< const StackGeomDet* >(det());
  const PixelGeomDetUnit* geomDetLower = dynamic_cast< const PixelGeomDetUnit* >(stackDet->lowerDet());
  return phase2clusterGlobalPos(geomDetLower, lowerCluster());
}

Global3DPoint VectorHit::upperGlobalPos() const {
  const StackGeomDet* stackDet = dynamic_cast< const StackGeomDet* >(det());
  const PixelGeomDetUnit* geomDetUpper = dynamic_cast< const PixelGeomDetUnit* >(stackDet->upperDet());
  return phase2clusterGlobalPos(geomDetUpper, upperCluster());
}

Global3DPoint VectorHit::phase2clusterGlobalPos(const PixelGeomDetUnit* geomDet, ClusterRef cluster) const {
  const PixelTopology * topo = &geomDet->specificTopology();
  float ix = cluster->center();
  float iy = cluster->column()+0.5; // halfway the column
  LocalPoint lp( topo->localX(ix), topo->localY(iy), 0 );          // x, y, z
  Global3DPoint gp = geomDet->surface().toGlobal(lp);
  return gp;
}

GlobalError VectorHit::lowerGlobalPosErr() const {
  const StackGeomDet* stackDet = dynamic_cast< const StackGeomDet* >(det());
  const PixelGeomDetUnit* geomDetLower = dynamic_cast< const PixelGeomDetUnit* >(stackDet->lowerDet());
  return phase2clusterGlobalPosErr(geomDetLower);
}

GlobalError VectorHit::upperGlobalPosErr() const {
  const StackGeomDet* stackDet = dynamic_cast< const StackGeomDet* >(det());
  const PixelGeomDetUnit* geomDetUpper = dynamic_cast< const PixelGeomDetUnit* >(stackDet->upperDet());
  return phase2clusterGlobalPosErr(geomDetUpper);
}

GlobalError VectorHit::phase2clusterGlobalPosErr(const PixelGeomDetUnit* geomDet) const {
  const PixelTopology * topo = &geomDet->specificTopology();
  float pitchX = topo->pitch().first;
  float pitchY = topo->pitch().second;
  LocalError le( pow(pitchX, 2) / 12., 0, pow(pitchY, 2) / 12.);               // e2_xx, e2_xy, e2_yy
  GlobalError ge( ErrorFrameTransformer().transform( le, geomDet->surface() ));
  return ge;
}

Global3DVector VectorHit::globalDelta() const {
  Local3DVector theLocalDelta = LocalVector(theDirection.x()*theDirection.z(), theDirection.y()*theDirection.z(), theDirection.z());
  Global3DVector g = det()->surface().toGlobal(theLocalDelta);
  return g;
}

Global3DVector VectorHit::globalDirection() const {
  return  (det()->surface().toGlobal(localDirection()));
}

std::pair<double,double> VectorHit::curvatureORphi(std::string curvORphi) const {

  double curvature = 0.0;
  double errorCurvature = 0.0;
  double phi = 0.0;

  //global pos and errors
  Global3DPoint gPositionLower = lowerGlobalPos();
  Global3DPoint gPositionUpper = upperGlobalPos();

  GlobalError gErrorLower = lowerGlobalPosErr();
  GlobalError gErrorUpper = upperGlobalPosErr();

  //insert lower and upper in the global sor
  if(gPositionLower.perp() > gPositionUpper.perp()){
    gPositionLower = upperGlobalPos();
    gPositionUpper = lowerGlobalPos();
    gErrorLower = upperGlobalPosErr();
    gErrorUpper = lowerGlobalPosErr();
  }

  double h1 = gPositionLower.x()*gPositionUpper.y() - gPositionUpper.x()*gPositionLower.y();

  //determine sign of curvature
  AlgebraicVector2 n1;
  n1[0] = -gPositionLower.y();
  n1[1] = gPositionLower.x();
  AlgebraicVector2 n2;
  n2[0] = gPositionUpper.x()-gPositionLower.x();
  n2[1] = gPositionUpper.y()-gPositionLower.y();

  double n3 = n1[0]*n2[0] + n1[1]*n2[1];
  double signCurv = -copysign(1.0,n3);
  double phi1 = atan2(gPositionUpper.y()-gPositionLower.y(),gPositionUpper.x()-gPositionLower.x());

  if(h1!=0) {
    double h2 = 2*h1;
    double r12 = pow(gPositionLower.x(),2) + pow(gPositionLower.y(),2);
    double r22 = pow(gPositionUpper.x(),2) + pow(gPositionUpper.y(),2);
    double h3 = (pow(gPositionLower.x(),2) - 2.*gPositionLower.x()*gPositionUpper.x() + pow(gPositionUpper.x(),2) + pow(gPositionLower.y(),2) - 2.*gPositionLower.y()*gPositionUpper.y() + pow(gPositionUpper.y(),2));
    double h4 = - pow(gPositionLower.x(),2)*gPositionUpper.x() + gPositionLower.x()*pow(gPositionUpper.x(),2) 
                + gPositionLower.x()*pow(gPositionUpper.y(),2) - gPositionUpper.x()*pow(gPositionLower.y(),2);
    double h5 = pow(gPositionLower.x(),2)*gPositionUpper.y() - pow(gPositionUpper.x(),2)*gPositionLower.y()
              + pow(gPositionLower.y(),2)*gPositionUpper.y() - gPositionLower.y()*pow(gPositionUpper.y(),2);

    //radius of circle
    double rho = sqrt(r12*r22*h3)/(2.*h1);
    curvature = 1./rho;

    //center of circle
    double xcentre = h5/h2;
    double ycentre = h4/h2;

    //to compute phi at the cluster points
    double xtg = gPositionLower.y() - ycentre;
    double ytg = -( gPositionLower.x() - xcentre);

    //to compute phi at the origin
    //double xtg = ycentre;
    //double ytg = -(xcentre);
    phi = atan2(ytg,xtg);

    AlgebraicROOTObject<4,4>::Matrix jacobian;
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 4; j++){
        jacobian[i][j] = 0.0;
      }
    }

    jacobian[0][0] = 1.0;    // dx1/dx1 dx1/dy1 dx2/dx1 dy2/dx1
    jacobian[1][1] = 1.0;    //dy1/dx1 dy1/dy1 dy2/dx1 dy2/dx1
    jacobian[2][0] = (h1*(2.*gPositionLower.x()*r22*h3 + (2.*gPositionLower.x() - 2.*gPositionUpper.x())*r12*r22))/(pow(r12*r22*h3,1.5)) 
                  - (2.*gPositionUpper.y())/sqrt(r12*r22*h3); // dkappa/dx1
    jacobian[2][1] = (2.*gPositionUpper.x())/sqrt(r12*r22*h3) + (h1*(2.*gPositionLower.y()*r22*h3 + r12*r22*(2.*gPositionLower.y() 
                  - 2.*gPositionUpper.y())))/pow(r12*r22*h3,1.5); // dkappa/dy1
    jacobian[2][2] = (2.*gPositionLower.y())/sqrt(r12*r22*h3) + (h1*(2.*gPositionUpper.x()*r12*h3 
                  - 2.*(gPositionLower.x() - gPositionUpper.x())*r12*r22))/pow(r12*r22*h3,1.5); // dkappa/dx2
    jacobian[2][3] = (h1*(2.*gPositionUpper.y()*r12*h3 - r12*r22*2.*(gPositionLower.y() - gPositionUpper.y())))/pow(r12*r22*h3,1.5)
                  - (2.*gPositionLower.x())/sqrt(r12*r22*h3); // dkappa/dy2

    for(int i = 0; i < 4; i++){
      jacobian[2][i] = -jacobian[2][i];
    }

    AlgebraicVector2 M;
    //to compute phi at the cluster points
    M[0] = (gPositionLower.y() - ycentre)/pow(rho,2); // dphi/dxcentre
    M[1] =-(gPositionLower.x() - xcentre)/pow(rho,2); // dphi/dycentre
    //to compute phi at the origin
    //float x0 = 0.0;
    //float y0 = 0.0;
    //M[0] = (y0 - ycentre)/pow(rho,2); // dphi/dxcentre
    //M[1] =-(x0 - xcentre)/pow(rho,2); // dphi/dycentre

    AlgebraicROOTObject<2,4>::Matrix K;
    K[0][0]=(2.*gPositionLower.x()*gPositionUpper.y())/h2 - (2.*gPositionUpper.y()*h5)/pow(h2,2); // dxm/dx1
    K[0][1]=(2.*gPositionUpper.x()*h5)/pow(h2,2) - (pow(gPositionUpper.x(),2) + pow(gPositionUpper.y(),2) - 2.*gPositionLower.y()*gPositionUpper.y())/h2; // dxm/dy1
    K[0][2]=(2.*gPositionLower.y()*h5)/pow(h2,2) - (2.*gPositionUpper.x()*gPositionLower.y())/h2; // dxm/dx2
    K[0][3]=(pow(gPositionLower.x(),2) + pow(gPositionLower.y(),2) - 2.*gPositionUpper.y()*gPositionLower.y())/h2 - (2.*gPositionLower.x()*h5)/pow(h2,2); // dxm/dy2
    K[1][0]=(pow(gPositionUpper.x(),2) - 2.*gPositionLower.x()*gPositionUpper.x() + pow(gPositionUpper.y(),2))/h2 - (2.*gPositionUpper.y()*h4)/pow(h2,2); // dym/dx1
    K[1][1]=(2.*gPositionUpper.x()*h4)/pow(h2,2) - (2.*gPositionUpper.x()*gPositionLower.y())/h2; // dym/dy1
    K[1][2]=(2.*gPositionLower.y()*h4)/pow(h2,2) - (pow(gPositionLower.x(),2) - 2.*gPositionUpper.x()*gPositionLower.x() + pow(gPositionLower.y(),2))/h2; // dym/dx2
    K[1][3]=(2.*gPositionLower.x()*gPositionUpper.y())/h2 - (2.*gPositionLower.x()*h4)/pow(h2,2); // dym/dy2

    AlgebraicVector4 N = M*K;
    jacobian[3][0] = N[0]; // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][1] = N[1]; // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][2] = N[2]; // dphi/(dx1,dy1,dx2,dy2)
    jacobian[3][3] = N[3]; // dphi/(dx1,dy1,dx2,dy2)

    //assign correct sign to the curvature errors
    if( (signCurv < 0 && curvature > 0 ) || (signCurv > 0 && curvature < 0 )){
      curvature=-curvature;
      for(int i = 0; i < 4; i++){
        jacobian[2][i] = -jacobian[2][i];
      }
    }

    // bring phi in the same quadrant as phi1
    if (abs(phi-phi1) > M_PI/2.){
      phi = phi+M_PI;
      if (phi>M_PI)
        phi=phi-2.*M_PI;
    }

    //computing the curvature error
    AlgebraicVector4 curvatureJacobian;
    for(int i = 0; i < 4; i++){
      curvatureJacobian[i] = jacobian[2][i];
    }

    AlgebraicROOTObject<4,4>::Matrix gErrors;
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 4; j++){
        gErrors[i][j] = 0.0;
      }
    }

    gErrors[0][0] = gErrorLower.cxx();
    gErrors[0][1] = gErrorLower.cyx();
    gErrors[1][0] = gErrorLower.cyx();
    gErrors[1][1] = gErrorLower.cyy();
    gErrors[2][2] = gErrorUpper.cxx();
    gErrors[2][3] = gErrorUpper.cyx();
    gErrors[3][2] = gErrorUpper.cyx();
    gErrors[3][3] = gErrorUpper.cyy();

    AlgebraicVector4 temp = curvatureJacobian;
    temp = temp*gErrors;
    errorCurvature = temp[0]*curvatureJacobian[0] + temp[1]*curvatureJacobian[1] + temp[2]*curvatureJacobian[2] + temp[3]*curvatureJacobian[3];
    //if(curvORphi == "curvature") std::cout << "curvature: " << curvature << std::endl;
    //if(curvORphi == "curvature") std::cout << "curvature error: " << errorCurvature << std::endl;


  } else {
std::cout << " straight line!" << std::endl;
    return std::make_pair(0.0,0.0);
  }
  
  if( curvORphi == "curvature" ) return std::make_pair(curvature,errorCurvature);
  else if( curvORphi == "phi"  ) return std::make_pair(phi,0.0);
  else return std::make_pair(0.0,0.0);

}

float VectorHit::transverseMomentum(const MagneticField* magField){

  GlobalPoint center(0.0, 0.0, 0.0);
  float magnT = magField->inTesla(center).mag();
  double rho = 1./curvatureORphi("curvature").first;
  //0.003 is because the curvature (rho) is in cm and not in m
  return (0.003*magnT*rho);

}

float VectorHit::momentum(const MagneticField* magField){
  return transverseMomentum(magField)/(1.*sin(theta()));
}

float VectorHit::theta(){
  return globalDirection().theta();
}

AlgebraicMatrix VectorHit::projectionMatrix() const {

//  static bool isInitialized=false;
  static AlgebraicMatrix the4DProjectionMatrix(4, 5, 0);
/*
  static AlgebraicMatrix the2DPhiProjMatrix(2, 5, 0);
  static AlgebraicMatrix the2DZProjMatrix(2, 5, 0);

  if (!isInitialized) {
    the4DProjectionMatrix[0][1] = 1;
    the4DProjectionMatrix[1][2] = 1;
    the4DProjectionMatrix[2][3] = 1;
    the4DProjectionMatrix[3][4] = 1;

    the2DPhiProjMatrix[0][1] = 1;
    the2DPhiProjMatrix[1][3] = 1;

    the2DZProjMatrix[0][2] = 1;
    the2DZProjMatrix[1][4] = 1;

    isInitialized= true;
  }

  if (dimension()==4) {
    return the4DProjectionMatrix;
  } else if (theProjection==phi) {
    return the2DPhiProjMatrix;
  } else if (theProjection==Z) {
    return the2DZProjMatrix;
  } else {
    return AlgebraicMatrix();
  }
*/ return the4DProjectionMatrix; //ERICA:QUESTO NON ESISTEVA!!
}


LocalError VectorHit::localPositionError() const {
  return LocalError(theCovMatrix[2][2],theCovMatrix[2][3],theCovMatrix[3][3]);
}


LocalError VectorHit::localDirectionError() const {
  return LocalError(theCovMatrix[0][0],theCovMatrix[0][1],theCovMatrix[1][1]);
}



AlgebraicSymMatrix VectorHit::parametersError() const {

  //think about a more efficient method
  AlgebraicSymMatrix result(4);
  for(int i = 0; i < theDimension; i++){
    for(int j = 0; j < theDimension; j++){
      result[i][j] = theCovMatrix[i][j];
    }
  }
  return result;
}

std::ostream& operator<<(std::ostream& os, const VectorHit& vh) {

  os << " VectorHit create in the DetId#: " << vh.geographicalId() << "\n" <<
        " Vectorhit local position      : " << vh.localPosition() << "\n" <<
        " Vectorhit local direction     : " << vh.localDirection() << "\n" <<
        " Vectorhit global direction    : " << vh.globalDirection() << "\n" <<
        //" Vectorhit theta               : " << vh.theta() << "\n" <<
        //" Cov: " << vh.parametersError() << "\n" <<
        //" Dim: " << vh.dimension() << "\n" <<
        //" chi2/ndof: " << vh.chi2() << "/" << vh.degreesOfFreedom() << "\n" <<
        " Lower cluster global position : " << vh.lowerGlobalPos() << "\n" <<
        " Upper cluster global position : " << vh.upperGlobalPos();

  return os;
}


/// Access to component RecHits (if any)
std::vector<const TrackingRecHit*> VectorHit::recHits() const{
  std::vector<const TrackingRecHit*> pointersOfRecHits;
  return pointersOfRecHits;
}


/// Non-const access to component RecHits (if any)
std::vector<TrackingRecHit*> VectorHit::recHits(){

  std::vector<TrackingRecHit*> pointersOfRecHits;
  return pointersOfRecHits;
}

