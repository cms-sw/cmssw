/**
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/15 17:25:28 $
 *  $Revision: 1.1 $
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFromRecHits.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Vector/interface/Pi.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//was
//#include "ClassReuse/GeomVector/interface/Pi.h"

#include "Geometry/Vector/interface/CoordinateSets.h"
//was
//#include "ClassReuse/GeomVector/interface/CoordinateSets.h"

#include "TrackingTools/DetLayers/interface/Enumerators.h"
//was
//#include "CommonDet/BasicDet/interface/Enumerators.h"

#include "Geometry/Surface/interface/BoundPlane.h"
//was
//#include "CommonDet/DetGeometry/interface/BoundPlane.h"

#include "Geometry/Surface/interface/RectangularPlaneBounds.h"
// was
//#include "CommonDet/DetGeometry/interface/RectangularPlaneBounds.h"

// Persistent version of a TrajectoryStateOnSurface
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

//>> what are now?
// #include "CommonDet/BasicDet/interface/Det.h"
// #include "CommonDet/BasicDet/interface/DetUnit.h"
// #include "CommonDet/BasicDet/interface/DetType.h"

// #include "CommonReco/Propagators/interface/BidirectionalPropagator.h"
// #include "CommonReco/GeaneInterface/interface/GeaneWrapper.h"

// #include "Muon/MUtilities/interface/MuonDebug.h"
// #include "Muon/MUtilities/interface/MuonDumper.h"
//<<

#include <iomanip>

MuonSeedFinder::MuonSeedFinder(){
  
  // FIXME put it in a pSet
  // theMinMomentum = pset.getParameter<double>("EndCapSeedMinPt");  //3.0
  theMinMomentum = 3.0;
  debug = true;
}


vector<TrajectorySeed> MuonSeedFinder::seeds(const edm::EventSetup& eSetup) const {

  //  MuonDumper debug;
  vector<TrajectorySeed> theSeeds;

  MuonSeedFromRecHits barrel;

  RecHitIterator iter;
  int num_bar = 0;
  for ( iter = theRhits.begin(); iter!= theRhits.end(); iter++ ){
    if ( (*iter)->isDT() ) {
      barrel.add(*iter);
      num_bar++;
    }
  }

  if ( num_bar ) {
    if ( debug ) // 5
      cout << "MuSeedGenByRecHits: Barrel Seeds " << num_bar << endl;
    theSeeds.push_back(barrel.seed(eSetup));
 
    //if ( debug ) //2
      // cout << theSeeds.back().startingState() << endl;
      // was
      // cout << theSeeds.back().freeTrajectoryState() << endl;
  }
  
  // 5
  if ( debug ) cout << "Endcap Seed" << endl;

  //Search ME1  ...
  MuonTransientTrackingRecHit *me1=0, *meit=0;
  float dPhiGloDir = .0;                            //  +v
  float bestdPhiGloDir = M_PI;                      //  +v
  int quality1 = 0, quality = 0;        //  +v  I= 5,6-p. / II= 4p.  / III= 3p.
  
  
  for ( iter = theRhits.begin(); iter!= theRhits.end(); iter++ ){
    if ( !(*iter)->isCSC() ) continue;

    // tmp compar. Glob-Dir for the same tr-segm:

    meit = *iter;

    int Nchi2 = 0;
    if ( meit->chi2()/(meit->recHits().size()*2.-4.) > 3. ) Nchi2 = 1;
    if ( meit->chi2()/(meit->recHits().size()*2.-4.) > 9. ) Nchi2 = 2;

    if ( meit->recHits().size() >  4 ) quality = 1 + Nchi2;
    if ( meit->recHits().size() == 4 ) quality = 3 + Nchi2;
    if ( meit->recHits().size() == 3 ) quality = 5 + Nchi2;

    dPhiGloDir = fabs ( meit->globalPosition().phi() - meit->globalDirection().phi() );
    if ( dPhiGloDir > M_PI   )  dPhiGloDir = 2.*M_PI - dPhiGloDir;  

    if ( dPhiGloDir > .2 ) quality = quality +1;  

    if ( !me1->isValid() ) {
      me1 = meit;
      quality1 = quality; 
      bestdPhiGloDir = dPhiGloDir;
    }

    if ( me1->isValid() && quality < quality1 ) {
        me1 = meit;
        quality1 = quality; 
        bestdPhiGloDir = dPhiGloDir;
    }    

    if ( me1->isValid() && bestdPhiGloDir > .03 ) {
      if ( dPhiGloDir < bestdPhiGloDir - .01 && quality == quality1 ) {
        me1 = meit;
        quality1 = quality; 
        bestdPhiGloDir = dPhiGloDir;
      }
    }    

  }   //  iter 


  if ( quality1 == 0 ) quality1 = 1;  

  bool good=false;

  if ( me1->isValid() ) {

    good = createEndcapSeed(me1, theSeeds); 
  
  }

    return theSeeds;
}

bool 
MuonSeedFinder::createEndcapSeed(MuonTransientTrackingRecHit *me, vector<TrajectorySeed>& theSeeds) const {

  bool result=false;
  //TODO: this is a mess!! I should use better the interface of
  //MuEndSegment (which must be improved!!!) 29-Mar-2001 SL

  // seed error = chamber dimension
  AlgebraicSymMatrix mat(5,0) ;
  mat[1][1] = (300./700.)*(300./700.)/12.;
  mat[2][2] = (300./700.)*(300./700.)/12.;
  mat[3][3] = 1.*1.;
  mat[4][4] = 6.*6.;

 // We want pT but it's not in RecHit interface, so we've put it in weight()
  float momentum = me->weight();

  // set minimum momentum for endcap seed
  float magmom = fabs( momentum );
  // maintain sign of momentum, for what it's worth...
  if ( magmom < theMinMomentum ) momentum = theMinMomentum * momentum/magmom;

  //big error for pt
  mat[0][0] = (0.25*0.25)/(momentum*momentum);
  LocalTrajectoryError error(mat);

  // LocalVector dir = me.localDirection();
  // float p_x = dir.x();
  // float p_y = dir.y();
  // float p_z = dir.z();
  // float this_z = p_z > 0 ? 1.: -1 ;
  // AlgebraicVector v = me.parameters();
  // float p_norm = momentum/sqrt(p_x*p_x + p_y*p_y);
  //LocalTrajectoryParameters param((1./p_norm),v[0],v[1],v[2],v[3], this_z);

  /// 12-Feb-2004 SL take phi from seg direction, eta from position
  LocalPoint segPos=me->localPosition();
  GlobalVector mom=me->globalPosition()-GlobalPoint();
  GlobalVector polar(GlobalVector::Spherical(mom.theta(),
                                             me->globalDirection().phi(),
                                             1.));
  polar *=fabs(momentum)/polar.perp();
  LocalVector segDirFromPos=me->det()->toLocal(polar);
  int charge=(int)(momentum/fabs(momentum));
  LocalTrajectoryParameters param(segPos,segDirFromPos, charge);

  // FIXME!!! The last two args are incorrect!
  MagneticField *magF=0;
  TrajectoryStateOnSurface tsos(param, error, me->det()->surface(), magF,me->weight() );
  //

  const FreeTrajectoryState state = *(tsos.freeState());

  //  MuonDumper debug;
  // 4
  //  if ( debug )  debug.dumpFTS(state);

  float z=0;
  /// magic number: eta=1.479 correspond to upper corner of ME1/1
  if (fabs(state.momentum().eta()) > 1.479) {
    /// plane at ME1/1
    z = 600.;
  } else {
    /// plane at ME1/2,3
    z = 696.;
  }
  if ( state.position().z() < 0 ) z = -z;

  Surface::PositionType pos(0., 0., z);
  Surface::RotationType rot;
  ReferenceCountingPointer<BoundPlane> 
    surface(new BoundPlane(pos, rot, RectangularPlaneBounds(720.,720.,1.)));

  Propagator* propagator= new SteppingHelixPropagator();

  TrajectoryStateOnSurface trj =  propagator->propagate( state, *surface );
  if ( trj.isValid() ) {
    const FreeTrajectoryState e_state = *trj.freeTrajectoryState();

    // FIXME
    //    TrajectorySeed seed(e_state, edm::OwnVector<TrackingRecHit>() ,oppositeToMomentum);
    TrajectorySeed seed;

    theSeeds.push_back(seed);

    //4
    if ( debug ) {
     cout<<"  Propag.oppositeToMomentum "<<endl;
     //    debug.dumpFTS(theSeeds.back().freeTrajectoryState());
     cout << "=== Successfull propagation" << endl;  // +v
    }
    result=true;
  } else {
    // 4
    if ( debug )  cout << "Invalid propagation" << endl;
    result=false;
  }
  delete propagator;
  return result;
}
