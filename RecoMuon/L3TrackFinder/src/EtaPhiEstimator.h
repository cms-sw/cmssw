#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

class EtaPhiEstimator : public Chi2MeasurementEstimatorBase {
 public:

  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of BoundPlane and maximalLocalDisplacement.
   */
  explicit EtaPhiEstimator(double eta, double phi,
			   const Chi2MeasurementEstimatorBase * estimator): 
    Chi2MeasurementEstimatorBase(estimator->chiSquaredCut(),
				 estimator->nSigmaCut()),
    estimator_(estimator),
    thedEta(eta),
    thedPhi(phi),
    thedEta2(eta*eta),
    thedPhi2(phi*phi)
      { }

    virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface& tsos,
					    const TrackingRecHit& aRecHit) const{
      
      std::pair<bool,double> primaryResult = estimator_->estimate(tsos,aRecHit);

      double dEta = fabs(tsos.globalPosition().eta() - aRecHit.globalPosition().eta());
      double dPhi = deltaPhi< double > (tsos.globalPosition().phi(), aRecHit.globalPosition().phi());

      double check = (dEta*dEta)/(thedEta2) + (dPhi*dPhi)/(thedPhi2);
      
      LogDebug("EtaPhiMeasurementEstimator")<< " The state to compare with is \n"<< tsos
					    << " The hit position is:\n" << aRecHit.globalPosition()
					    << " deta: "<< dEta<< " dPhi: "<<dPhi<<" check: "<<check<<" primaryly: "<< primaryResult.second;

      if (check <= 1)
	//      if (dEta < thedEta && dPhi <thedPhi)
	return std::make_pair(true, primaryResult.second);
      else
	return std::make_pair(false, primaryResult.second);
    }
    
    virtual EtaPhiEstimator* clone() const {
      return new EtaPhiEstimator(*this);
    }

 private:
    const Chi2MeasurementEstimatorBase * estimator_;
    double thedEta,thedPhi,thedEta2,thedPhi2;
};
