/** \class Chi2ChargeMeasurementEstimatorESProducer
 *  ESProducer for Chi2ChargeMeasurementEstimator.
 *
 *  \author speer
 */

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/MeasurementDet/interface/ClusterFilterPayload.h"

#include <limits>

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

namespace {

  class Chi2ChargeMeasurementEstimator final : public Chi2MeasurementEstimator {
  public:
    /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of Plane and maximalLocalDisplacement.
   */
    template <typename... Args>
    Chi2ChargeMeasurementEstimator(float minGoodPixelCharge,
                                   float minGoodStripCharge,
                                   float pTChargeCutThreshold,
                                   Args&&... args)
        : Chi2MeasurementEstimator(args...),
          minGoodPixelCharge_(minGoodPixelCharge),
          minGoodStripCharge_(minGoodStripCharge),
          pTChargeCutThreshold2_(pTChargeCutThreshold >= 0. ? pTChargeCutThreshold * pTChargeCutThreshold
                                                            : std::numeric_limits<float>::max()) {}

    bool preFilter(const TrajectoryStateOnSurface& ts, const MeasurementEstimator::OpaquePayload& opay) const override;

    Chi2ChargeMeasurementEstimator* clone() const override { return new Chi2ChargeMeasurementEstimator(*this); }

  private:
    const float minGoodPixelCharge_;
    const float minGoodStripCharge_;
    const float pTChargeCutThreshold2_;

    bool checkClusterCharge(DetId id, SiStripCluster const& cluster, const TrajectoryStateOnSurface& ts) const {
      return siStripClusterTools::chargePerCM(id, cluster, ts.localParameters()) > minGoodStripCharge_;
    }
  };

  bool Chi2ChargeMeasurementEstimator::preFilter(const TrajectoryStateOnSurface& ts,
                                                 const MeasurementEstimator::OpaquePayload& opay) const {
    // what we got?
    if (opay.tag != ClusterFilterPayload::myTag)
      return true;  // not mine...

    auto const& clf = reinterpret_cast<ClusterFilterPayload const&>(opay);

    if (ts.globalMomentum().perp2() > pTChargeCutThreshold2_)
      return true;

    DetId detid = clf.detId;
    uint32_t subdet = detid.subdetId();

    if (subdet > 2) {
      return checkClusterCharge(detid, *clf.cluster[0], ts) &&
             (nullptr == clf.cluster[1] || checkClusterCharge(detid, *clf.cluster[1], ts));
    }

    /*  pixel charge not implemented as not used...
     auto const & thit = static_cast<const SiPixelRecHit &>(hit);
     thit.cluster()->charge() ...

  */

    return true;
  }

}  // namespace

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include <memory>

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorParams.h"

namespace {

  class Chi2ChargeMeasurementEstimatorESProducer : public edm::ESProducer {
  public:
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    Chi2ChargeMeasurementEstimatorESProducer(const edm::ParameterSet& p);
    ~Chi2ChargeMeasurementEstimatorESProducer() override;
    std::unique_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord&);

  private:
    const double m_maxChi2;
    const double m_nSigma;
    const double m_maxDis;
    const double m_maxSag;
    const double m_minTol;
    const double m_minpt;
    const float m_minGoodStripCharge;
    const double m_pTChargeCutThreshold;
  };

  void Chi2ChargeMeasurementEstimatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    auto desc = chi2MeasurementEstimatorParams::getFilledConfigurationDescription();
    desc.add<std::string>("ComponentName", "Chi2Charge");
    desc.add<double>("pTChargeCutThreshold", -1.);
    edm::ParameterSetDescription descCCC = getConfigurationDescription4CCC(CCC::kLoose);
    desc.add<edm::ParameterSetDescription>("clusterChargeCut", descCCC);
    descriptions.add("Chi2ChargeMeasurementEstimatorDefault", desc);
  }

  Chi2ChargeMeasurementEstimatorESProducer::Chi2ChargeMeasurementEstimatorESProducer(const edm::ParameterSet& pset)
      : m_maxChi2(pset.getParameter<double>("MaxChi2")),
        m_nSigma(pset.getParameter<double>("nSigma")),
        m_maxDis(pset.getParameter<double>("MaxDisplacement")),
        m_maxSag(pset.getParameter<double>("MaxSagitta")),
        m_minTol(pset.getParameter<double>("MinimalTolerance")),
        m_minpt(pset.getParameter<double>("MinPtForHitRecoveryInGluedDet")),
        m_minGoodStripCharge(clusterChargeCut(pset)),
        m_pTChargeCutThreshold(pset.getParameter<double>("pTChargeCutThreshold")) {
    std::string const& myname = pset.getParameter<std::string>("ComponentName");
    setWhatProduced(this, myname);
  }

  Chi2ChargeMeasurementEstimatorESProducer::~Chi2ChargeMeasurementEstimatorESProducer() {}

  std::unique_ptr<Chi2MeasurementEstimatorBase> Chi2ChargeMeasurementEstimatorESProducer::produce(
      const TrackingComponentsRecord& iRecord) {
    auto minGoodPixelCharge = 0;

    return std::make_unique<Chi2ChargeMeasurementEstimator>(minGoodPixelCharge,
                                                            m_minGoodStripCharge,
                                                            m_pTChargeCutThreshold,
                                                            m_maxChi2,
                                                            m_nSigma,
                                                            m_maxDis,
                                                            m_maxSag,
                                                            m_minTol,
                                                            m_minpt);
  }

}  // namespace

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(Chi2ChargeMeasurementEstimatorESProducer);
