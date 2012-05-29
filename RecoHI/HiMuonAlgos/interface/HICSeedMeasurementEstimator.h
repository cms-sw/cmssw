#ifndef HIC_SEED_MeasurementEstimator_HIC_H 
#define HIC_SEED_MeasurementEstimator_HIC_H 

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

//#define HICMEASUREMENT_DEBUG
namespace cms {
class HICSeedMeasurementEstimator:public MeasurementEstimator {
public:

explicit HICSeedMeasurementEstimator(bool& trust,int nsig):trtrue(trust), theNSigma(nsig) {}

virtual MeasurementEstimator::HitReturnType  estimate( const TrajectoryStateOnSurface& ts, 
                                     const TransientTrackingRecHit& hit) const;
				     
virtual MeasurementEstimator::SurfaceReturnType estimate( const TrajectoryStateOnSurface& ts, 
                            const BoundPlane& plane) const;
			    
virtual MeasurementEstimator::Local2DVector 
  maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
			    const BoundPlane& plane) const;

double nSigmaCut() const {return theNSigma;}

void set(double& phi, double& z) {thePhi=phi;theZ=z;}

double getZ() {return theZ;}

double getPhi() {return thePhi;}

HICSeedMeasurementEstimator* clone() const {
    return new HICSeedMeasurementEstimator(*this);
}
 		 
private:

double thePhi;
double theZ;
bool   trtrue;
int    theNSigma;
			   			   
};
}
#endif // HIC_SEED_MeasurementEstimator_HIC_H
