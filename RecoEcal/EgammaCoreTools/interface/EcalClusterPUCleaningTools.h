#ifndef RecoEcal_EgammaCoreTools_EcalClusterPUCleaningTools_h
#define RecoEcal_EgammaCoreTools_EcalClusterPUCleaningTools_h

/** \class EcalClusterPUCleaningTools
 *  
 * tool to clean reco::Supercluster from effects of multiple interactions ( PU )
 *
 * \author G. Franzoni - UMN
 * 
 * \version $Id: 
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"

class CaloGeometry;

class EcalClusterPUCleaningTools {
 public:
  EcalClusterPUCleaningTools( const edm::Event &ev, const edm::EventSetup &es, const edm::InputTag& redEBRecHits, const edm::InputTag& redEERecHits );
  ~EcalClusterPUCleaningTools();
  reco::SuperCluster CleanedSuperCluster(float xi, const reco::SuperCluster &cluster, const edm::Event &ev);
  
 private:
  void getGeometry( const edm::EventSetup &es );
  void getEBRecHits( const edm::Event &ev, const edm::InputTag& redEBRecHits );
  void getEERecHits( const edm::Event &ev, const edm::InputTag& redEERecHits );
  
  const CaloGeometry *geometry_;
  const EcalRecHitCollection *ebRecHits_;
  const EcalRecHitCollection *eeRecHits_;
		
};

#endif
