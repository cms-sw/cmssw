#ifndef RecoEcal_EcalClusterAlgos_LogPositionCalc_h
#define RecoEcal_EcalClusterAlgos_LogPositionCalc_h

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

// CMSSW headers from this subsystem
#include "DataFormats/Math/interface/Point3D.h"
// C/C++ headers
#include <string>
#include <vector>
typedef math::XYZPoint Point;
Point getECALposition(std::vector<reco::EcalRecHitData> recHits, const CaloSubdetectorGeometry & geometry);//Position determination



#endif
