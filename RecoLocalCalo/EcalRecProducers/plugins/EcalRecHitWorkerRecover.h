#ifndef RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerRecover_hh
#define RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerRecover_hh

/** \class EcalRecHitWorkerRecover
  *  Algorithms to recover dead channels
  *
  */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include <vector>

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/EcalBarrelGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryAlgos.h"

class EcalRecHitWorkerRecover : public EcalRecHitWorkerBaseClass {
public:
  EcalRecHitWorkerRecover(const edm::ParameterSet&, edm::ConsumesCollector& c);
  ~EcalRecHitWorkerRecover() override{};

  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt, const EcalUncalibratedRecHit& uncalibRH, EcalRecHitCollection& result) override;

protected:
  void insertRecHit(const EcalRecHit& hit, EcalRecHitCollection& collection);
  float recCheckCalib(float energy, int ieta);
  bool alreadyInserted(const DetId& id);
  float estimateEnergy(int ieta,
                       EcalRecHitCollection* hits,
                       const std::set<DetId>& sId,
                       const std::vector<DetId>& vId,
                       const EcalTPGScale& tpgscale);
  bool checkChannelStatus(const DetId& id, const std::vector<int>& statusestoexclude);

  edm::ESHandle<EcalLaserDbService> laser;
  edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> laserToken_;

  // isolated dead channels
  edm::ESHandle<CaloTopology> caloTopology_;
  edm::ESHandle<CaloGeometry> caloGeometry_;
  edm::ESHandle<EcalChannelStatus> chStatus_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopologyToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> chStatusToken_;

  double singleRecoveryThreshold_;
  double sum8RecoveryThreshold_;
  std::string singleRecoveryMethod_;
  bool killDeadChannels_;

  bool recoverEBIsolatedChannels_;
  bool recoverEEIsolatedChannels_;
  bool recoverEBVFE_;
  bool recoverEEVFE_;
  bool recoverEBFE_;
  bool recoverEEFE_;

  // list of channel statuses for which recovery in EE should
  // not be attempted
  std::vector<int> dbStatusToBeExcludedEE_;
  std::vector<int> dbStatusToBeExcludedEB_;

  const edm::EventSetup* eventSetup_ = nullptr;
  // dead FE
  EcalTPGScale::Tokens ecalScaleTokens_;
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> tpDigiToken_;
  edm::ESHandle<EcalElectronicsMapping> pEcalMapping_;
  const EcalElectronicsMapping* ecalMapping_;
  double logWarningEtThreshold_EB_FE_;
  double logWarningEtThreshold_EE_FE_;

  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap_;

  edm::ESHandle<CaloSubdetectorGeometry> pEBGeom_;
  const CaloSubdetectorGeometry* ebGeom_;
  const CaloGeometry* geo_;
  edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> pEcalMappingToken_;
  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> ttMapToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometryRecord> pEBGeomToken_;
  std::unique_ptr<EcalRecHitSimpleAlgo> rechitMaker_;

  std::set<DetId> recoveredDetIds_EB_;
  std::set<DetId> recoveredDetIds_EE_;

  EcalTPGScale::Tokens tpgscaleTokens_;

  EcalDeadChannelRecoveryAlgos<EBDetId> ebDeadChannelCorrector;
  EcalDeadChannelRecoveryAlgos<EEDetId> eeDeadChannelCorrector;
};

#endif
