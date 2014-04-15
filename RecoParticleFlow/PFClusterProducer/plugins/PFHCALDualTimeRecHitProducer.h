#ifndef RecoParticleFlow_PFClusterProducer_PFHCALDualTimeRecHitProducer_h_
#define RecoParticleFlow_PFClusterProducer_PFHCALDualTimeRecHitProducer_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducer.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

/**\class PFHCALDualTimeRecHitProducer
\brief Producer for particle flow rechits  (PFRecHit) in HCAL Upgrade

\author Chris Tully
\date   June 2012
*/

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;


class PFHCALDualTimeRecHitProducer : public PFRecHitProducer {
 public:
  explicit PFHCALDualTimeRecHitProducer(const edm::ParameterSet&);
  ~PFHCALDualTimeRecHitProducer();
 
 private:


  /// gets hcal barrel and endcap rechits, 
  /// translate them to PFRecHits, which are stored in the rechits vector
  void createRecHits(std::vector<reco::PFRecHit>& rechits,
		     std::vector<reco::PFRecHit>& rechitsCleaned,
		     edm::Event&, const edm::EventSetup&);




  reco::PFRecHit*  createHcalRecHit( const DetId& detid, 
				     double energy,
				     PFLayer::Layer layer,
				     const CaloSubdetectorGeometry* geom,
				     unsigned newDetId=0);
  

  

  /// find and set the neighbours to a given rechit
  /// this works for ecal, hcal, ps
  void 
    findRecHitNeighbours( reco::PFRecHit& rh, 
			  const std::map<unsigned,unsigned >& sortedHits, 
			  const CaloSubdetectorTopology& barrelTopo,
			  const CaloSubdetectorGeometry& barrelGeom, 
			  const CaloSubdetectorTopology& endcapTopo,
			  const CaloSubdetectorGeometry& endcapGeom );
  
  /// find and set the neighbours to a given rechit
  /// this works for hcal CaloTowers. 
  /// Should be possible to have a single function for all detectors
  void 
    findRecHitNeighboursCT( reco::PFRecHit& rh, 
			    const std::map<unsigned,unsigned >& sortedHits,
			    const CaloSubdetectorTopology& topology );
  
  DetId getNorth(const DetId& id, const CaloSubdetectorTopology& topology);
  DetId getSouth(const DetId& id, const CaloSubdetectorTopology& topology);
  

  // ----------member data ---------------------------
  
  // ----------access to event data
  edm::InputTag    inputTagHcalRecHitsHBHE_;
  edm::InputTag    inputTagHcalRecHitsHF_;
  edm::InputTag    inputTagCaloTowers_;
  
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

  // Compensate for dead ECAL channels
  bool ECAL_Compensate_;
  double ECAL_Threshold_;
  double ECAL_Compensation_;
  unsigned int ECAL_Dead_Code_;

  // Depth correction for EM and HAD rechits in the HF
  double EM_Depth_;
  double HAD_Depth_;

};

#endif
