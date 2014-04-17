#ifndef RecoParticleFlow_PFClusterProducer_PFCTRecHitProducer_h_
#define RecoParticleFlow_PFClusterProducer_PFCTRecHitProducer_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "CondFormats//HcalObjects/interface/HcalPFCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"


class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;

class PFCTRecHitProducer : public edm::stream::EDProducer<> {
 public:
  explicit PFCTRecHitProducer(const edm::ParameterSet&);
  ~PFCTRecHitProducer();

  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, 
				    const edm::EventSetup & es) override;
  
  void produce(edm::Event& iEvent, 
	       const edm::EventSetup& iSetup);


  reco::PFRecHit*  createHcalRecHit( const DetId& detid, 
				     double energy,
				     PFLayer::Layer layer,
				     const CaloSubdetectorGeometry* geom,
				     const CaloTowerDetId& newDetId);


 protected:
  double  thresh_Barrel_;
  double  thresh_Endcap_;
  const HcalChannelQuality* theHcalChStatus;
  const EcalChannelStatus* theEcalChStatus;
  const CaloTowerConstituentsMap* theTowerConstituentsMap;


  // ----------access to event data
  edm::EDGetTokenT<HBHERecHitCollection> hcalToken_;
  edm::EDGetTokenT<HFRecHitCollection> hfToken_;
  edm::EDGetTokenT<CaloTowerCollection> towersToken_;
  
  /// threshold for HF
  double           thresh_HF_;
  // Navigation in HF:  False = no real clustering in HF; True  = do clustering 
  bool   navigation_HF_;
  double weight_HFem_;
  double weight_HFhad_;

  // Apply HCAL DPG rechit calibration
  bool HCAL_Calib_;
  bool HF_Calib_;
  float HCAL_Calib_29;
  float HF_Calib_29;

  // Don't allow large energy in short fibres if there is no energy in long fibres
  double shortFibre_Cut;  
  double longFibre_Fraction;

  // Don't allow large energy in long fibres if there is no energy in short fibres
  double longFibre_Cut;  
  double shortFibre_Fraction;

  // Also apply HCAL DPG cleaning
  bool applyLongShortDPG_;

  // Don't allow too large timing excursion if energy in long/short fibres is large enough
  double longShortFibre_Cut;  
  double minShortTiming_Cut;
  double maxShortTiming_Cut;
  double minLongTiming_Cut;
  double maxLongTiming_Cut;

  bool applyTimeDPG_;
  bool applyPulseDPG_;
  int  HcalMaxAllowedHFLongShortSev_;
  int  HcalMaxAllowedHFDigiTimeSev_;
  int  HcalMaxAllowedHFInTimeWindowSev_;
  int  HcalMaxAllowedChannelStatusSev_;

  int hcalHFLongShortFlagValue_;
  int hcalHFDigiTimeFlagValue_;
  int hcalHFInTimeWindowFlagValue_;

  // Compensate for dead ECAL channels
  bool ECAL_Compensate_;
  double ECAL_Threshold_;
  double ECAL_Compensation_;
  unsigned int ECAL_Dead_Code_;

  // Depth correction for EM and HAD rechits in the HF
  double EM_Depth_;
  double HAD_Depth_;


  int m_maxDepthHB;
  int m_maxDepthHE;

  PFRecHitNavigatorBase* navigator_;


};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCTRecHitProducer);

#endif
