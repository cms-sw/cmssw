#ifndef DTSEGTORPC_H
#define DTSEGTORPC_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

class RPCGeometry;
class DTGeometry;
class DTObjectMap;
class MuonGeometryRecord;

class DTSegtoRPC {
public:
  explicit DTSegtoRPC(edm::ConsumesCollector iC, const edm::ParameterSet&);
  std::unique_ptr<RPCRecHitCollection> thePoints(DTRecSegment4DCollection const* all4DSegments,
                                                 edm::EventSetup const& iSetup,
                                                 bool debug,
                                                 double eyr);

private:
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeoToken_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeoToken_;
  edm::ESGetToken<DTObjectMap, MuonGeometryRecord> dtMapToken_;

  bool incldt;
  bool incldtMB4;
  double MinCosAng;
  double MaxD;
  double MaxDrb4;
  double MaxDistanceBetweenSegments;
  int minPhiBX;
  int maxPhiBX;
};

#endif
