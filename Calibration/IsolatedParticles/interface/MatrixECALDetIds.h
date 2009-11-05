#ifndef CalibrationIsolatedParticleseMatrizECALDetIds_h
#define CalibrationIsolatedParticleseMatrizECALDetIds_h

#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

namespace spr{

  std::vector<DetId> matrixECALIds(const DetId& det, int ieta, int iphi, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> matrixECALIds(const DetId& det, int ietaE, int ietaW, int iphiN, int iphiS, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> newECALIdNS(std::vector<DetId>& dets, unsigned int last, int ieta, int iphi, const CaloDirection& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> newECALIdNS(std::vector<DetId>& dets, unsigned int last, int ietaE, int ietaW, int iphiN, int iphiS, const CaloDirection& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> newECALIdEW(std::vector<DetId>& dets, unsigned int last, int ieta, const CaloDirection& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> newECALIdEW(std::vector<DetId>& dets, unsigned int last, int ietaE, int ietaW, const CaloDirection& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::pair<DetId,bool> simpleMove(DetId& det, const CaloDirection& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);
}

#endif
