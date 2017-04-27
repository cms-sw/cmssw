#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerSimple_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerSimple_hh

/** \class HGCalRecHitSimpleAlgo
  *  Simple algoritm to make HGCAL rechits from HGCAL uncalibrated rechits
  *
  *  \author Valeri Andreev
  */

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalRecHitSimpleAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

class HGCalRecHitWorkerSimple : public HGCalRecHitWorkerBaseClass {
 public:
  HGCalRecHitWorkerSimple(const edm::ParameterSet&);
  virtual ~HGCalRecHitWorkerSimple();                       
  
  void set(const edm::EventSetup& es);
  bool run(const edm::Event& evt, const HGCUncalibratedRecHit& uncalibRH, HGCRecHitCollection & result);
  
 protected:
  
  double HGCEE_keV2DIGI_,  hgceeUncalib2GeV_;
  std::vector<double> HGCEE_fCPerMIP_;
  double HGCHEF_keV2DIGI_, hgchefUncalib2GeV_;
  std::vector<double> HGCHEF_fCPerMIP_;
  double HGCHEB_keV2DIGI_, hgchebUncalib2GeV_;
  bool HGCEE_isSiFE_, HGCHEF_isSiFE_, HGCHEB_isSiFE_;
  


  std::vector<double> HGCEE_noise_fC_;
  std::vector<double> HGCHEF_noise_fC_;
  double HGCHEB_noise_MIP_;


  std::array<const HGCalDDDConstants*, 3> ddds_;
  
  std::vector<int> v_chstatus_;
  
  std::vector<int> v_DB_reco_flags_;
  bool killDeadChannels_;

  std::vector<double> rcorr_;
  std::vector<float> weights_;
  std::unique_ptr<HGCalRecHitSimpleAlgo> rechitMaker_;
  std::unique_ptr<hgcal::RecHitTools> tools_;

  const bool isRealistic_ = false;
  float nSigmaThreshold_;


};

#endif
