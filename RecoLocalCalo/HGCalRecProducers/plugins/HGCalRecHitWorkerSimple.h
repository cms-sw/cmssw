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
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

class HGCalRecHitWorkerSimple : public HGCalRecHitWorkerBaseClass {
public:
  HGCalRecHitWorkerSimple(const edm::ParameterSet&);
  ~HGCalRecHitWorkerSimple() override;

  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt, const HGCUncalibratedRecHit& uncalibRH, HGCRecHitCollection& result) override;

protected:
  enum detectortype { hgcee = 1, hgcfh = 2, hgcbh = 3, hgchfnose = 4 };

  double hgcEE_keV2DIGI_, hgceeUncalib2GeV_;
  std::vector<double> hgcEE_fCPerMIP_;
  std::vector<double> hgcEE_cce_;
  double hgcHEF_keV2DIGI_, hgchefUncalib2GeV_;
  std::vector<double> hgcHEF_fCPerMIP_;
  std::vector<double> hgcHEF_cce_;
  double hgcHEB_keV2DIGI_, hgchebUncalib2GeV_;
  double hgcHFNose_keV2DIGI_, hgchfnoseUncalib2GeV_;
  std::vector<double> hgcHFNose_fCPerMIP_;
  std::vector<double> hgcHFNose_cce_;
  bool hgcEE_isSiFE_, hgcHEF_isSiFE_, hgcHEB_isSiFE_, hgcHFNose_isSiFE_;

  std::vector<double> hgcEE_noise_fC_;
  std::vector<double> hgcHEF_noise_fC_;
  std::vector<double> hgcHFNose_noise_fC_;
  double hgcHEB_noise_MIP_;

  std::array<const HGCalDDDConstants*, 4> ddds_;

  std::vector<int> v_chstatus_;

  std::vector<int> v_DB_reco_flags_;
  bool killDeadChannels_;

  uint32_t rangeMatch_;
  uint32_t rangeMask_;

  std::vector<double> rcorr_, rcorrNose_;
  std::vector<float> weights_, weightsNose_;
  std::unique_ptr<HGCalRecHitSimpleAlgo> rechitMaker_;
  std::unique_ptr<hgcal::RecHitTools> tools_;

  hgcalsimclustertime::ComputeClusterTime timeEstimatorSi_;
};

#endif
