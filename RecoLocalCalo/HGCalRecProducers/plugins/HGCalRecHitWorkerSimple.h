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

class HGCalRecHitWorkerSimple : public HGCalRecHitWorkerBaseClass {
 public:
  HGCalRecHitWorkerSimple(const edm::ParameterSet&);
  virtual ~HGCalRecHitWorkerSimple();                       
  
  void set(const edm::EventSetup& es);
  bool run(const edm::Event& evt, const HGCUncalibratedRecHit& uncalibRH, HGCRecHitCollection & result);
  
 protected:
  
  double HGCEE_keV2DIGI_,  hgceeUncalib2GeV_;
  double HGCHEF_keV2DIGI_, hgchefUncalib2GeV_;
  double HGCHEB_keV2DIGI_, hgchebUncalib2GeV_;
  
  std::vector<int> v_chstatus_;
  
  std::vector<int> v_DB_reco_flags_;
  
  bool killDeadChannels_;
  
  std::unique_ptr<HGCalRecHitSimpleAlgo> rechitMaker_;
};

#endif
