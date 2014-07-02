//CMSSW headers
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

//FAMOS headers
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

ParticlePropagator::ParticlePropagator() : 
  BaseParticlePropagator(), random(0) {;}

ParticlePropagator::ParticlePropagator(const RawParticle& myPart,
				       double RCyl, double ZCyl, 
				       const MagneticFieldMap* aFieldMap,
				       const RandomEngineAndDistribution* engine) :
  BaseParticlePropagator(myPart,RCyl,ZCyl,0.),
  theFieldMap(aFieldMap),
  random(engine)
{
  setMagneticField(fieldMap(X(),Y(),Z()));
  initProperDecayTime();
}

ParticlePropagator::ParticlePropagator( const RawParticle& myPart,
					const MagneticFieldMap* aFieldMap,
					const RandomEngineAndDistribution* engine) :
  BaseParticlePropagator(myPart,0.,0.,0.),
  theFieldMap(aFieldMap),
  random(engine)
 
{
  setMagneticField(fieldMap(X(),Y(),Z()));
  initProperDecayTime();
}

ParticlePropagator::ParticlePropagator(const XYZTLorentzVector& mom, 
				       const XYZTLorentzVector& vert, float q,
				       const MagneticFieldMap* aFieldMap) :
  BaseParticlePropagator(RawParticle(mom,vert),0.,0.,0.),
  theFieldMap(aFieldMap),
  random(0)
{
  setCharge(q);
  setMagneticField(fieldMap(X(),Y(),Z()));
}

ParticlePropagator::ParticlePropagator(const XYZTLorentzVector& mom, 
				       const XYZVector& vert, float q,
				       const MagneticFieldMap* aFieldMap) :
  BaseParticlePropagator(
    RawParticle(mom,XYZTLorentzVector(vert.X(),vert.Y(),vert.Z(),0.0)),0.,0.,0.),
  theFieldMap(aFieldMap),
  random(0)
{
  setCharge(q);
  setMagneticField(fieldMap(X(),Y(),Z()));
}

ParticlePropagator::ParticlePropagator(const FSimTrack& simTrack,
				       const MagneticFieldMap* aFieldMap,
				       const RandomEngineAndDistribution* engine) :
  BaseParticlePropagator(RawParticle(simTrack.type(),simTrack.momentum()),
			 0.,0.,0.),
  theFieldMap(aFieldMap),
  random(engine)
{
  setVertex(simTrack.vertex().position());
  setMagneticField(fieldMap(X(),Y(),Z()));
  if ( simTrack.decayTime() < 0. ) { 
    if ( simTrack.nDaughters() ) 
      // This particle already decayed, don't decay it twice
      this->setProperDecayTime(1E99);
    else
      // This particle hasn't decayed yet. Decay time according to particle lifetime
      initProperDecayTime();
  } else {
    // Decay time pre-defined at generator level 
    this->setProperDecayTime(simTrack.decayTime());
  }
}

ParticlePropagator::ParticlePropagator(const ParticlePropagator& myPropPart) :
  BaseParticlePropagator(myPropPart),
  theFieldMap(myPropPart.theFieldMap)
{  
  //  setMagneticField(fieldMap(x(),y(),z()));
}

ParticlePropagator::ParticlePropagator(const BaseParticlePropagator& myPropPart,
				       const MagneticFieldMap* aFieldMap) :
  BaseParticlePropagator(myPropPart),
  theFieldMap(aFieldMap)
{  
  setMagneticField(fieldMap(X(),Y(),Z()));
}


void 
ParticlePropagator::initProperDecayTime() {

  // And this is the proper time at which the particle will decay
  double properDecayTime = 
    (pid()==0||pid()==22||abs(pid())==11||abs(pid())==2112||abs(pid())==2212||
     !random) ?
    1E99 : -PDGcTau() * std::log(random->flatShoot());

  this->setProperDecayTime(properDecayTime);

}


bool
ParticlePropagator::propagateToClosestApproach(double x0, double y0, bool first) {
  setMagneticField(fieldMap(0.,0.,0.));
  return BaseParticlePropagator::propagateToClosestApproach(x0,y0,first);
}

bool
ParticlePropagator::propagateToNominalVertex(const XYZTLorentzVector& v) {
  setMagneticField(fieldMap(0.,0.,0.));
  return BaseParticlePropagator::propagateToNominalVertex(v);
}

ParticlePropagator
ParticlePropagator::propagated() const {
  return ParticlePropagator(BaseParticlePropagator::propagated(),theFieldMap);
}

double
ParticlePropagator::fieldMap(double xx,double yy, double zz) {
  // Arguments now passed in cm.
  //  return MagneticFieldMap::instance()->inTesla(GlobalPoint(xx/10.,yy/10.,zz/10.)).z();
  // Return a dummy value for neutral particles!
  return charge() == 0.0 || theFieldMap == 0 ? 
    4. : theFieldMap->inTeslaZ(GlobalPoint(xx,yy,zz));
}

double
ParticlePropagator::fieldMap(const TrackerLayer& layer, double coord, int success) {
  // Arguments now passed in cm.
  //  return MagneticFieldMap::instance()->inTesla(GlobalPoint(xx/10.,yy/10.,zz/10.)).z();
  // Return a dummy value for neutral particles!
  return charge() == 0.0 || theFieldMap == 0 ? 
    4. : theFieldMap->inTeslaZ(layer,coord,success);
}

bool
ParticlePropagator::propagateToBoundSurface(const TrackerLayer& layer) {


  fiducial = true;
  BoundDisk const* disk = layer.disk();
  //  bool disk = layer.forward();
  //  double innerradius=-999;
  double innerradius = disk ? layer.diskInnerRadius() : -999. ;

  //  if( disk ) {
    //    const Surface& surface = layer.surface();
    //    const BoundDisk & myDisk = dynamic_cast<const BoundDisk&>(surface);
    //    innerradius=myDisk.innerRadius();	  
  //    innerradius=myDisk->innerRadius();	  
  //  }

  bool done = propagate();

  // Set the magnetic field at the new location (if succesfully propagated)
  if ( done && !hasDecayed() ) { 
    if ( success == 2 ) 
      setMagneticField(fieldMap(layer,r(),success));
    else if ( success == 1 )
      setMagneticField(fieldMap(layer,z(),success));
  }	
       
  // There is some real material here
  fiducial = !(!disk &&  success!=1) &&
	     !( disk && (success!=2  || r()<innerradius));

  return done;
}

void 
ParticlePropagator::setPropagationConditions(const TrackerLayer& layer, 
					     bool firstLoop) { 
  // Set the magentic field
  // setMagneticField(fieldMap(x(),y(),z()));

  // Set R and Z according to the Tracker Layer characteristics.
  //  const Surface& surface = layer.surface();

  if( layer.forward() ) {

    //    const BoundDisk & myDisk = dynamic_cast<const BoundDisk&>(surface);
    // ParticlePropagator works in mm, whereas the detector geometry is in cm
    BaseParticlePropagator::setPropagationConditions(
                                  layer.diskOuterRadius(),
				  fabs(layer.disk()->position().z()),
				  firstLoop);       

  // ... or if it is a cylinder barrel 
  } else {

    //    const BoundCylinder & myCylinder = dynamic_cast<const BoundCylinder &>(surface);
    // ParticlePropagator works now in cm
    BaseParticlePropagator::setPropagationConditions(
					 layer.cylinder()->bounds().width()/2.,
					 layer.cylinder()->bounds().length()/2.,
					 firstLoop);
  }

}

