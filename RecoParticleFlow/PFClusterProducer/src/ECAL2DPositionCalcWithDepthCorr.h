#ifndef __ECAL2DPositionCalcWithDepthCorr_H__
#define __ECAL2DPositionCalcWithDepthCorr_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoParticleFlow/PFClusterProducer/interface/ECALRecHitResolutionProvider.h"

/// This is EGM version of the ECAL position + depth correction calculation
class ECAL2DPositionCalcWithDepthCorr : public PFCPositionCalculatorBase {
 public:
  ECAL2DPositionCalcWithDepthCorr(const edm::ParameterSet& conf) :
    PFCPositionCalculatorBase(conf), 
    _param_T0_EB(conf.getParameter<double>("T0_EB")),
    _param_T0_EE(conf.getParameter<double>("T0_EE")),
    _param_T0_ES(conf.getParameter<double>("T0_ES")),
    _param_W0(conf.getParameter<double>("W0")),
    _param_X0(conf.getParameter<double>("X0")),
    _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization")),
    _caloGeom(NULL),
    _ebGeom(NULL),
    _eeGeom(NULL),
    _esGeom(NULL),
    _esPlus(false),
    _esMinus(false) {
        _timeResolutionCalc.reset(NULL);
    if( conf.exists("timeResolutionCalc") ) {
      const edm::ParameterSet& timeResConf = 
        conf.getParameterSet("timeResolutionCalc");
        _timeResolutionCalc.reset(new ECALRecHitResolutionProvider(timeResConf)); 
      }
    }
  ECAL2DPositionCalcWithDepthCorr(const ECAL2DPositionCalcWithDepthCorr&) = delete;
  ECAL2DPositionCalcWithDepthCorr& operator=(const ECAL2DPositionCalcWithDepthCorr&) = delete;

  void update(const edm::EventSetup& es);

  void calculateAndSetPosition(reco::PFCluster&);
  void calculateAndSetPositions(reco::PFClusterCollection&);

 private:  
  const double _param_T0_EB;
  const double _param_T0_EE;
  const double _param_T0_ES;
  const double _param_W0;
  const double _param_X0;
  const double _minAllowedNorm;

  
  const CaloGeometryRecord* _caloGeom;
  const CaloSubdetectorGeometry* _ebGeom;
  const CaloSubdetectorGeometry* _eeGeom;
  const CaloSubdetectorGeometry* _esGeom;
  bool _esPlus, _esMinus;

  std::unique_ptr<ECALRecHitResolutionProvider> _timeResolutionCalc;
  
  void calculateAndSetPositionActual(reco::PFCluster&) const;
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory,
		  ECAL2DPositionCalcWithDepthCorr,
		  "ECAL2DPositionCalcWithDepthCorr");

#endif
