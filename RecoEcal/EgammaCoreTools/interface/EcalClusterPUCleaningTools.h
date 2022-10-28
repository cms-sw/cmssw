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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class CaloGeometry;

class EcalClusterPUCleaningTools {
public:
  EcalClusterPUCleaningTools(edm::ConsumesCollector &cc,
                             const edm::InputTag &redEBRecHits,
                             const edm::InputTag &redEERecHits);
  ~EcalClusterPUCleaningTools();
  reco::SuperCluster CleanedSuperCluster(float xi,
                                         const reco::SuperCluster &cluster,
                                         const edm::Event &ev,
                                         const edm::EventSetup &es);

private:
  void getEBRecHits(const edm::Event &ev);
  void getEERecHits(const edm::Event &ev);

  const edm::EDGetTokenT<EcalRecHitCollection> pEBRecHitsToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> pEERecHitsToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;

  const EcalRecHitCollection *ebRecHits_;
  const EcalRecHitCollection *eeRecHits_;
};

#endif
