// Class Header
#include "RecoMuon/L2MuonSeedGenerator/src/L2MuonSeedGenerator.h"

//Service Records
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"

// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"

// C++
#include <vector>
#include "CLHEP/Vector/ThreeVector.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Surface/interface/Surface.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "Geometry/Surface/interface/RectangularPlaneBounds.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

using namespace std;
using namespace edm;
using namespace l1extra ;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L2MuonSeedGenerator::L2MuonSeedGenerator(const edm::ParameterSet& iConfig) : 
  theL1MinPt(iConfig.getParameter<double>("L1MinPt")),
  theL1MaxEta(iConfig.getParameter<double>("L1MaxEta")),
  theL1MinQuality(iConfig.getParameter<double>("L1MinQuality")),
  source_( iConfig.getParameter< edm::InputTag >( "src" ) )
{
  produces<TrajectorySeedCollection>(); 
  
}


L2MuonSeedGenerator::~L2MuonSeedGenerator()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L2MuonSeedGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());

  edm::ESHandle<MagneticField> theMagField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);
  
  // Muon particles
  edm::Handle< L1MuonParticleCollection > muColl ;
  iEvent.getByLabel( source_, muColl ) ;
  cout << "Number of muons " << muColl->size() << endl ;
    
  //
  // GMT Trigger
  //
  
  float radius = 513.;
  
  L1MuonParticleCollection::const_iterator it;
  for(it=muColl->begin(); it!=muColl->end(); it++) {
    float pt    =  (*it).pt();
    float eta   =  (*it).eta();
    float theta =  2*atan(exp(-eta));
    float phi   =  (*it).phi();      
    int charge  =  (*it).charge();
    
    const L1MuGMTCand *tmpCand = (*it).gmtMuonCand();
    unsigned int quality =  tmpCand->quality();
    
    unsigned int det;
    if (quality == 7) // matched ?
      det =  isFwd() ? 5 : 3;
    else 
      det =  isRPC() ? 1 : ( isFwd()? 4 : 2); 
    
    if ( pt < theL1MinPt || fabs(eta) > theL1MaxEta ) continue;

    if ( quality < (unsigned int)theL1MinQuality ) continue;
    
    bool barrel = ( det == 4 || det == 5 ) ? false : true;
    if ( det == 1 && fabs(eta)>1 ) barrel = false;
    
    if ( !barrel ) radius = 800.;
    
    if (  barrel && pt < 3.5 ) pt = 3.5;
    if ( !barrel && pt < 1.0 ) pt = 1.0;
    
    Hep3Vector vec(0.,1.,0.);
    vec.setTheta(theta);
    vec.setPhi(phi);
    if (  barrel ) radius = fabs(radius/sin(theta));
    if ( !barrel ) radius = fabs(radius/cos(theta));
    vec.setMag(radius);
    
    GlobalPoint pos(vec.x(),vec.y(),vec.z());
    
    float x = cos(phi)*sin(theta);
    float y = sin(phi)*sin(theta);
    float z = cos(theta);
    
    GlobalVector mom3(x, y, z);
    float mom = pt/sqrt(x*x + y*y);
    mom3 = mom3*mom;
    
    GlobalTrajectoryParameters param(pos,mom3,charge,&*theMagField);
    AlgebraicSymMatrix mat(5,0);
    
    mat[0][0] = (0.25/pt)*(0.25/pt);  // sigma^2(charge/abs_momentum)
    if ( !barrel ) mat[0][0] = (0.4/pt)*(0.4/pt);
    
    mat[1][1] = 0.05*0.05;        // sigma^2(lambda)
    mat[2][2] = 0.2*0.2;          // sigma^2(phi)
    mat[3][3] = 20.*20.;          // sigma^2(x_transverse))
    mat[4][4] = 20.*20.;          // sigma^2(y_transverse))
    
    CurvilinearTrajectoryError error(mat);
    GlobalVector dir = mom3.unit();
    
    double wx = dir.x();
    double wy = dir.y();
    double wz = dir.z();
    double dydx = wy / wx;
    double uy = 1. / sqrt(1 + dydx * dydx);
    double ux = - dydx * uy;
    Surface::RotationType rot(    ux,     uy,              0,
				  -wz*uy,  wz*ux,  wx*uy - wy*ux,
				  wx,     wy,             wz);
    
    BoundPlane* bPlane = new BoundPlane(pos, rot, RectangularPlaneBounds(1e10,1e10,1e10));
    TrajectoryStateOnSurface basic(param, error, *bPlane);
    
    const FreeTrajectoryState state = *(basic.freeTrajectoryState());
    TrajectoryStateOnSurface trj;
    SteppingHelixPropagator prop(&*theMagField,oppositeToMomentum);
    if ( barrel ) {    
      pos = GlobalPoint(0., 0., 0.);
      Surface::RotationType rot2;
      ReferenceCountingPointer<BoundCylinder> cyl(new BoundCylinder( pos, rot2, SimpleCylinderBounds(399.,401.,-1200.,1200.)));
      trj = prop.propagate(state, *cyl);
    }
    else {
      float z = -500.;
      if ( state.position().z() > 0 ) z = 500.;
      pos = GlobalPoint(0., 0., z);
      Surface::RotationType rot2;
      ReferenceCountingPointer<BoundPlane> surface(new BoundPlane(pos, rot2, RectangularPlaneBounds(720.,720.,1.)));
      trj = prop.propagate(state, *surface);
    }
    
    if ( trj.isValid() ) {
      const FreeTrajectoryState e_state = *trj.freeTrajectoryState();
      // Transform it in a TrajectoryStateOnSurface
      TrajectoryStateTransform tsTransform;
      PTrajectoryStateOnDet *seedTSOS = tsTransform.persistentState( basic , det);
      //<< FIXME:
      // TrajectorySeed theSeed(e_state, rechitcontainer,oppositeToMomentum);
      // But is:
      edm::OwnVector<TrackingRecHit> container;
      TrajectorySeed* seed = new TrajectorySeed(*seedTSOS,container,alongMomentum);
      // is this right?

      output->push_back(*seed);
    } 

  }//end loop over extended candidates
 
  
  iEvent.put(output);
}
// ------------ method called once each job just before starting event loop  ------------
void 
L2MuonSeedGenerator::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L2MuonSeedGenerator::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L2MuonSeedGenerator)
