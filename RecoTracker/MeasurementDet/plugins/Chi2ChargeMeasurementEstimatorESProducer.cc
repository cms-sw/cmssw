/** \class Chi2ChargeMeasurementEstimatorESProducer
 *  ESProducer for Chi2ChargeMeasurementEstimator.
 *
 *  \author speer
 */



#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/MeasurementDet/interface/ClusterFilterPayload.h"

#include<limits>

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"


namespace {

class Chi2ChargeMeasurementEstimator GCC11_FINAL : public Chi2MeasurementEstimator {
public:

  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of Plane and maximalLocalDisplacement.
   */
  explicit Chi2ChargeMeasurementEstimator(double maxChi2, double nSigma,
	float minGoodPixelCharge, float minGoodStripCharge,
	float pTChargeCutThreshold) : 
    Chi2MeasurementEstimator( maxChi2, nSigma),
    minGoodPixelCharge_(minGoodPixelCharge),
    minGoodStripCharge_(minGoodStripCharge) {
      if (pTChargeCutThreshold>=0.) pTChargeCutThreshold2_=pTChargeCutThreshold*pTChargeCutThreshold;
      else pTChargeCutThreshold2_=std::numeric_limits<float>::max();
    }


  bool preFilter(const TrajectoryStateOnSurface& ts,
                 const MeasurementEstimator::OpaquePayload  & opay) const override;


  virtual Chi2ChargeMeasurementEstimator* clone() const {
    return new Chi2ChargeMeasurementEstimator(*this);
  }
private:

  float minGoodPixelCharge_; 
  float minGoodStripCharge_;
  float pTChargeCutThreshold2_;

  bool checkClusterCharge(DetId id, SiStripCluster const & cluster, const TrajectoryStateOnSurface& ts) const {
    return siStripClusterTools::chargePerCM(id, cluster, ts.localParameters() ) >  minGoodStripCharge_;

  }


};

bool Chi2ChargeMeasurementEstimator::preFilter(const TrajectoryStateOnSurface& ts,
                                               const MeasurementEstimator::OpaquePayload  & opay) const {

  // what we got?
  if (opay.tag != ClusterFilterPayload::myTag) return true;  // not mine...
  
  auto const & clf = reinterpret_cast<ClusterFilterPayload const &>(opay);

  if (ts.globalMomentum().perp2()>pTChargeCutThreshold2_) return true;

  DetId detid = clf.detId;
  uint32_t subdet = detid.subdetId();

  if (subdet>2) {
    return checkClusterCharge(detid, *clf.cluster[0],ts) && ( nullptr==clf.cluster[1] || checkClusterCharge(detid, *clf.cluster[1],ts) ) ; 

  }

  /*  pixel charge not implemented as not used...
     auto const & thit = static_cast<const SiPixelRecHit &>(hit);
     thit.cluster()->charge() ...

  */

  return true;
}

}



#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include <boost/shared_ptr.hpp>

namespace {


class  Chi2ChargeMeasurementEstimatorESProducer: public edm::ESProducer{
 public:
  Chi2ChargeMeasurementEstimatorESProducer(const edm::ParameterSet & p);
  virtual ~Chi2ChargeMeasurementEstimatorESProducer(); 
  boost::shared_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<Chi2MeasurementEstimatorBase> _estimator;

  double maxChi2_;
  double nSigma_;
  float minGoodPixelCharge_; 
  float minGoodStripCharge_;
  float pTChargeCutThreshold_;

};

Chi2ChargeMeasurementEstimatorESProducer::Chi2ChargeMeasurementEstimatorESProducer(const edm::ParameterSet & pset) 
{
  std::string const & myname = pset.getParameter<std::string>("ComponentName");
  setWhatProduced(this,myname);

  maxChi2_             = pset.getParameter<double>("MaxChi2");
  nSigma_              = pset.getParameter<double>("nSigma");
  minGoodPixelCharge_  = 0;
  minGoodStripCharge_  = clusterChargeCut(pset);
  pTChargeCutThreshold_= pset.getParameter<double>("pTChargeCutThreshold");
  

}

Chi2ChargeMeasurementEstimatorESProducer::~Chi2ChargeMeasurementEstimatorESProducer() {}

boost::shared_ptr<Chi2MeasurementEstimatorBase> 
Chi2ChargeMeasurementEstimatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  _estimator = boost::shared_ptr<Chi2MeasurementEstimatorBase>(
	new Chi2ChargeMeasurementEstimator(maxChi2_,nSigma_, 
		minGoodPixelCharge_, minGoodStripCharge_, pTChargeCutThreshold_));
  return _estimator;
}

}


#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(Chi2ChargeMeasurementEstimatorESProducer);

