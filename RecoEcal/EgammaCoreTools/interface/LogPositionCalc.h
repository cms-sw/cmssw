#ifndef RecoEcal_EgammaCoreTools_LogPositionCalc_h
#define RecoEcal_EgammaCoreTools_LogPositionCalc_h

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/Math/interface/Point3D.h"
// C/C++ headers
#include <string>
#include <vector>

class LogPositionCalc

{
 public:
  LogPositionCalc();

  virtual ~LogPositionCalc();

  typedef math::XYZPoint Point;

  static Point getECALposition(std::vector<reco::EcalRecHitData> recHits, const CaloSubdetectorGeometry);

};

#endif
