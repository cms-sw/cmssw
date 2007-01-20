/**
 *  See header file for a description of this class.
 *
 *  $Date: 2006/08/28 16:16:59 $
 *  $Revision: 1.15 $
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author porting  R. Bellan
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFromRecHits.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Vector/interface/Pi.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Vector/interface/CoordinateSets.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/Surface/interface/RectangularPlaneBounds.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iomanip>

using namespace std;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

MuonSeedFinder::MuonSeedFinder(){
  
  // FIXME put it in a pSet
  // theMinMomentum = pset.getParameter<double>("EndCapSeedMinPt");  //3.0
  theMinMomentum = 3.0;
}


vector<TrajectorySeed> MuonSeedFinder::seeds(const edm::EventSetup& eSetup) const {

  const std::string metname = "Muon|RecoMuon|MuonSeedFinder";

  //  MuonDumper debug;
  vector<TrajectorySeed> theSeeds;

  MuonSeedFromRecHits barrel;

  int num_bar = 0;
  for ( MuonRecHitContainer::const_iterator iter = theRhits.begin(); iter!= theRhits.end(); iter++ ){
    if ( (*iter)->isDT() ) {
      barrel.add(*iter);
      num_bar++;
    }
  }

  if ( num_bar ) {
    LogDebug(metname)
      << "Barrel Seeds " << num_bar << endl;
    theSeeds.push_back(barrel.seed(eSetup));
 
    //if ( debug ) //2
      // cout << theSeeds.back().startingState() << endl;
      // was
      // cout << theSeeds.back().freeTrajectoryState() << endl;
  }
  
  // 5
  else LogDebug(metname) << "Endcap Seed" << endl;

  //Search ME1  ...
  MuonRecHitPointer me1=0, meit=0;
  float dPhiGloDir = .0;                            //  +v
  float bestdPhiGloDir = M_PI;                      //  +v
  int quality1 = 0, quality = 0;        //  +v  I= 5,6-p. / II= 4p.  / III= 3p.
  
  
  for ( MuonRecHitContainer::const_iterator iter = theRhits.begin(); iter!= theRhits.end(); iter++ ){
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

    if(!me1){
      me1 = meit;
      quality1 = quality; 
      bestdPhiGloDir = dPhiGloDir;
    }
    
    if(me1) {
      
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
    }
    
  }   //  iter 
  

  if ( quality1 == 0 ) quality1 = 1;  

  bool good=false;

  if(me1)
    if ( me1->isValid() )
      good = createEndcapSeed(me1, theSeeds,eSetup); 
  
  return theSeeds;
}

bool 
MuonSeedFinder::createEndcapSeed(MuonRecHitPointer last, 
				 vector<TrajectorySeed>& theSeeds,
				 const edm::EventSetup& eSetup) const {

  const std::string metname = "Muon|RecoMuon|MuonSeedFinder";
  
  edm::ESHandle<MagneticField> field;
  eSetup.get<IdealMagneticFieldRecord>().get(field);
  
  AlgebraicSymMatrix mat(5,0) ;

  // this perform H.T() * parErr * H, which is the projection of the 
  // the measurement error (rechit rf) to the state error (TSOS rf)
  // Legenda:
  // H => is the 4x5 projection matrix
  // parError the 4x4 parameter error matrix of the RecHit
  
  mat = last->parametersError().similarityT( last->projectionMatrix() );
  
  // We want pT but it's not in RecHit interface, so we've put it within this class
  float momentum = computePt(last,&*field);
  // FIXME
  //  float smomentum = 0.25*momentum; // FIXME!!!!
  float smomentum = 25; 

  MuonSeedFromRecHits seedCreator;
  TrajectorySeed cscSeed = seedCreator.createSeed(momentum,smomentum,last,eSetup);

  theSeeds.push_back(cscSeed);

  // FIXME
  return true;
}


float MuonSeedFinder::computePt(ConstMuonRecHitPointer muon, const MagneticField *field) const {
// assume dZ = dPhi*R*C, here C = pZ/pT
// =======================================================================
// ptc: I suspect the following comment should really be
// dZ/dPsi = 0.5*dz/dPhi
// which I can derive if I assume the particle has travelled in a circle
// projected onto the global xy plane, starting at the origin on the z-axis.
// Here Psi is the angle traced out in the xy plane by the projection of the
// helical path of the charged particle. The axis of the helix is assumed 
// parallel to the main B field of the solenoid.
// =======================================================================
// dZ/dPhi = 0.5*dZ/dPsi, here phi = atan2(y,x), psi = rho*s

// ptc: If the local direction is effectively (0,0,1) or (0,0,-1)
// then it's ridiculous to follow this algorithm... just set some
// arbitrary 'high' value and note the sign is undetermined

//@@ DO SOMETHING SANE WITH THESE TRAP VALUES
  static float small = 1.e-06;
  static float big = 1.e+10;

  LocalVector lod = muon->localDirection();
  if ( fabs(lod.x())<small && fabs(lod.y())<small ) {
    return big;
  }

  GlobalPoint gp = muon->globalPosition();
  GlobalVector gv = muon->globalDirection();
  float getx0 = gp.x();
  float getay = gv.y()/gv.z();
  float gety0 = gp.y();
  float getax = gv.x()/gv.z();
  float getz0 = gp.z();
  
  float dZdPhi = 0.5f*gp.perp2()/(getx0*getay - gety0*getax);
  float dZdT = getz0/gp.perp();
  float rho = dZdT/dZdPhi;
  
  // convert to pT (watch the sign !)
  GlobalVector fld = field->inInverseGeV( gp );
  return -fld.z()/rho;
}

bool 
MuonSeedFinder::createEndcapSeed_OLD(MuonRecHitPointer me, 
				 vector<TrajectorySeed>& theSeeds,
				 const edm::EventSetup& eSetup) const {

  const std::string metname = "Muon|RecoMuon|MuonSeedFinder";
  
  edm::ESHandle<MagneticField> field;
  eSetup.get<IdealMagneticFieldRecord>().get(field);
  
  bool result=false;
  //TODO: this is a mess!! I should use better the interface of
  //MuEndSegment (which must be improved!!!) 29-Mar-2001 SL

  // seed error = chamber dimension
  AlgebraicSymMatrix mat(5,0) ;

  // mat[1][1] = (300./700.)*(300./700.)/12.;
  // mat[2][2] = (300./700.)*(300./700.)/12.;
  // mat[3][3] = 1.*1.;
  // mat[4][4] = 6.*6.;

  // this perform H.T() * parErr * H, which is the projection of the 
  // the measurement error (rechit rf) to the state error (TSOS rf)
  // Legenda:
  // H => is the 4x5 projection matrix
  // parError the 4x4 parameter error matrix of the RecHit
  
  // FIXME Use this!!!!!!!!
  mat = me->parametersError().similarityT( me->projectionMatrix() );
  
  // We want pT but it's not in RecHit interface, so we've put it within this class
  float momentum = computePt(me,&*field);

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

  TrajectoryStateOnSurface tsos(param, error, me->det()->surface(), &*field);

  const FreeTrajectoryState state = *(tsos.freeState());

  MuonPatternRecoDumper debugDumper;
  LogDebug(metname) << debugDumper.dumpFTS(state);

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

  // FIXME must get prop from ESSetup
  // FIXME NCA: NCA: use anyDirection for the time being
  //  Propagator* propagator= new SteppingHelixPropagator(&*field,oppositeToMomentum);
  Propagator* propagator= new SteppingHelixPropagator(&*field,anyDirection);

  TrajectoryStateOnSurface trj =  propagator->propagate( state, *surface );
  if ( trj.isValid() ) {
    const FreeTrajectoryState e_state = *trj.freeTrajectoryState();

    // Transform it in a TrajectoryStateOnSurface
    TrajectoryStateTransform tsTransform;

    // FIXME FIXME TEST
//     PTrajectoryStateOnDet *seedTSOS =
//       tsTransform.persistentState( trj ,me->geographicalId().rawId());
    
    // FIXME the tsos is defined on the "me" surface, this must be changed!!!
    PTrajectoryStateOnDet *seedTSOS =
      tsTransform.persistentState( tsos ,me->geographicalId().rawId());


    //<< FIXME would be:
    
    // TrajectorySeed theSeed(e_state, rechitcontainer,oppositeToMomentum);
    // But is:
    edm::OwnVector<TrackingRecHit> container;
    // container.push_back(me->hit()->clone() ); 

    TrajectorySeed seed(*seedTSOS,container,oppositeToMomentum);
    //>> is it right??

    theSeeds.push_back(seed);
    
    LogDebug(metname)<<"  Propag.oppositeToMomentum "<<endl;
    LogDebug(metname)<< debugDumper.dumpTSOS(trj);
    LogDebug(metname) << "=== Successfull propagation" << endl;  // +v
    
    result=true;
  } else {
    LogDebug(metname) << "Invalid propagation" << endl;
    result=false;
  }
  delete propagator;
  return result;
}
