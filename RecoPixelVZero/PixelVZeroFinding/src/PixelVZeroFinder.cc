#include "RecoPixelVZero/PixelVZeroFinding/interface/PixelVZeroFinder.h"

#include "Geometry/Vector/interface/GlobalVector.h"

/*****************************************************************************/
PixelVZeroFinder::PixelVZeroFinder
  (const edm::EventSetup& es,
   const edm::ParameterSet& pset)
{
  // Get track-pair level cuts
  maxDcaR = pset.getParameter<double>("maxDcaR");
  maxDcaZ = pset.getParameter<double>("maxDcaZ");

  // Get mother cuts
  minCrossingRadius = pset.getParameter<double>("minCrossingRadius");
  maxCrossingRadius = pset.getParameter<double>("maxCrossingRadius");
  maxImpactMother   = pset.getParameter<double>("maxImpactMother");

  // Get magnetic field
  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  theMagField = magField.product();

  // Get closest approach finder
  theApproach = new ClosestApproachInRPhi();
}

/*****************************************************************************/
PixelVZeroFinder::~PixelVZeroFinder()
{
  delete theApproach;
}

/*****************************************************************************/
FreeTrajectoryState PixelVZeroFinder::getTrajectory(const reco::Track& track)
{ 
  GlobalPoint position(track.vertex().x(),
                       track.vertex().y(),
                       track.vertex().z());

  GlobalVector momentum(track.momentum().x(),
                        track.momentum().y(),
                        track.momentum().z());

  GlobalTrajectoryParameters gtp(position,momentum,
                                 track.charge(),theMagField);
  
  FreeTrajectoryState fts(gtp);

  return fts; 
} 

/*****************************************************************************/
bool PixelVZeroFinder::checkTrackPair(const reco::Track& posTrack,
                                      const reco::Track& negTrack,
                                      reco::VZeroData& data)
{
  // Get trajectories
  FreeTrajectoryState posTraj = getTrajectory(posTrack);
  FreeTrajectoryState negTraj = getTrajectory(negTrack);

  // Closest approach
  pair<GlobalTrajectoryParameters, GlobalTrajectoryParameters>  
    gtp = theApproach->trajectoryParameters(posTraj,negTraj);

  // Closest points
  pair<GlobalPoint, GlobalPoint>
    points(gtp.first.position(), gtp.second.position());

  // Momenta at closest point
  pair<GlobalVector,GlobalVector>
    momenta(gtp.first.momentum(), gtp.second.momentum());

  // dcaR
  float dcaR = (points.first - points.second).perp();
  GlobalPoint crossing(0.5*(points.first.x() + points.second.x()),
                       0.5*(points.first.y() + points.second.y()),
                       0.5*(points.first.z() + points.second.z()));

  if(dcaR < maxDcaR &&
     crossing.perp() > minCrossingRadius &&
     crossing.perp() < maxCrossingRadius)
  {
    // dcaZ
    float theta = 0.5*(posTrack.theta() + negTrack.theta());
    float dcaZ = fabs((points.first - points.second).z()) * sin(theta); 

    if(dcaZ < maxDcaZ)
    {
      // Momentum of the mother
      GlobalVector momentum = momenta.first + momenta.second;

      // Impact parameter of the mother in the plane
      GlobalVector r_(crossing.x(),crossing.y(),0);
      GlobalVector p_(momentum.x(),momentum.y(),0);

      GlobalVector b_ = r_ - (r_*p_)*p_ / p_.mag2();
      float impact = b_.mag(); 

      if(impact < maxImpactMother)
      {
        // Armenteros variables
        float pt =
          (momenta.first.cross(momenta.second)).mag()/momentum.mag();
        float alpha =
          (momenta.first.mag2() - momenta.second.mag2())/momentum.mag2();

        // Fill data
        data.dcaR = dcaR;
        data.dcaZ = dcaZ;
        data.crossingPoint = crossing;
        data.impactMother  = impact;
        data.armenterosPt    = pt;
        data.armenterosAlpha = alpha;

        return true;
      }
    }
  }

  return false;
}

