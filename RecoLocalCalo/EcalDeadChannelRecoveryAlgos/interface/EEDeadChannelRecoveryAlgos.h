#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EEDeadChannelRecoveryAlgos_HH
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EEDeadChannelRecoveryAlgos_HH
 
/** \class EEDeadChannelRecoveryAlgos
  *
  *  \author Stilianos Kesisoglou - Institute of Nuclear and Particle Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
  */
 

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapHardcodedTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <string>


class EEDeadChannelRecoveryAlgos  {
   public:
  EEDeadChannelRecoveryAlgos(const CaloTopology * theCaloTopology);
 
  EcalRecHit correct(const EEDetId Id, const EcalRecHitCollection* hit_collection, std::string algo_, double Sum8Cut, bool* AccFlag);
  
 private:
  
  
  const CaloTopology * calotopo;
  double MakeNxNMatrice_RelMC(EEDetId itID,const EcalRecHitCollection* hit_collection, double *MNxN_RelMC, bool* AccFlag);
  double MakeNxNMatrice_RelDC(EEDetId itID,const EcalRecHitCollection* hit_collection, double *MNxN_RelDC, bool* AccFlag);
  
  // ----------member data ---------------------------
  
};
#endif
