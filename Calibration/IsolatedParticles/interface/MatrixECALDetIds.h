#ifndef CalibrationIsolatedParticleseMatrizECALDetIds_h
#define CalibrationIsolatedParticleseMatrizECALDetIds_h

#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

namespace spr{

  std::vector<DetId> matrixECALIds(const DetId& det, int ieta, int iphi, const CaloGeometry* geo, const CaloTopology* caloTopology, bool debug=false);

  std::vector<DetId> matrixECALIds(const DetId& det, int ietaE, int ietaW, int iphiN, int iphiS, const CaloGeometry* geo, const CaloTopology* caloTopology, bool debug=false);

  std::vector<DetId> newECALIdNS(std::vector<DetId>& dets, unsigned int last, int ieta, int iphi, std::vector<CaloDirection>& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> newECALIdNS(std::vector<DetId>& dets, unsigned int last, std::vector<int>& ietaE, std::vector<int>& ietaW, std::vector<int>& iphiN, std::vector<int>& iphiS, std::vector<CaloDirection>& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> newECALIdEW(std::vector<DetId>& dets, unsigned int last, int ieta, std::vector<CaloDirection>& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> newECALIdEW(std::vector<DetId>& dets, unsigned int last, std::vector<int>& ietaE, std::vector<int>& ietaW, std::vector<CaloDirection>& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

  std::vector<DetId> simpleMove(DetId& det, const CaloDirection& dir, const CaloSubdetectorTopology& barrelTopo, const CaloSubdetectorTopology& endcapTopo, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, int& flag, bool debug=false);

  std::vector<DetId> extraIds(const DetId& det, std::vector<DetId>& dets, int ietaE, int ietaW, int iphiN, int iphiS, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug=false);

}
#endif
