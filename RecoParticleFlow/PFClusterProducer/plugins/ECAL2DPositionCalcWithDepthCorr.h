#ifndef __ECAL2DPositionCalcWithDepthCorr_H__
#define __ECAL2DPositionCalcWithDepthCorr_H__

#include <memory>

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoParticleFlow/PFClusterProducer/interface/CaloRecHitResolutionProvider.h"

/// This is EGM version of the ECAL position + depth correction calculation
class ECAL2DPositionCalcWithDepthCorr : public PFCPositionCalculatorBase {
public:
  ECAL2DPositionCalcWithDepthCorr(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : PFCPositionCalculatorBase(conf, cc),
        _param_T0_EB(conf.getParameter<double>("T0_EB")),
        _param_T0_EE(conf.getParameter<double>("T0_EE")),
        _param_T0_ES(conf.getParameter<double>("T0_ES")),
        _param_W0(conf.getParameter<double>("W0")),
        _param_X0(conf.getParameter<double>("X0")),
        _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization")),
        _ebGeom(nullptr),
        _eeGeom(nullptr),
        _esGeom(nullptr),
        _esPlus(false),
        _esMinus(false),
        _geomToken(cc.esConsumes<edm::Transition::BeginLuminosityBlock>()) {
    _timeResolutionCalc.reset(nullptr);
    if (conf.exists("timeResolutionCalc")) {
      const edm::ParameterSet& timeResConf = conf.getParameterSet("timeResolutionCalc");
      _timeResolutionCalc = std::make_unique<CaloRecHitResolutionProvider>(timeResConf);
    }
  }
  ECAL2DPositionCalcWithDepthCorr(const ECAL2DPositionCalcWithDepthCorr&) = delete;
  ECAL2DPositionCalcWithDepthCorr& operator=(const ECAL2DPositionCalcWithDepthCorr&) = delete;

  void update(const edm::EventSetup& es) override;

  void calculateAndSetPosition(reco::PFCluster&) override;
  void calculateAndSetPositions(reco::PFClusterCollection&) override;

private:
  const double _param_T0_EB;
  const double _param_T0_EE;
  const double _param_T0_ES;
  const double _param_W0;
  const double _param_X0;
  const double _minAllowedNorm;

  //const CaloGeometryRecord  _caloGeom;
  const CaloSubdetectorGeometry* _ebGeom;
  const CaloSubdetectorGeometry* _eeGeom;
  const CaloSubdetectorGeometry* _esGeom;
  bool _esPlus, _esMinus;

  std::unique_ptr<CaloRecHitResolutionProvider> _timeResolutionCalc;

  void calculateAndSetPositionActual(reco::PFCluster&) const;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> _geomToken;
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory, ECAL2DPositionCalcWithDepthCorr, "ECAL2DPositionCalcWithDepthCorr");

#endif
