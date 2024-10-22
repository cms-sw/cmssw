#ifndef CSCSEGTORPC_H
#define CSCSEGTORPC_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

class RPCGeometry;
class CSCGeometry;
class CSCObjectMap;
class MuonGeometryRecord;

class CSCSegtoRPC {
public:
  explicit CSCSegtoRPC(edm::ConsumesCollector iC, const edm::ParameterSet&);
  std::unique_ptr<RPCRecHitCollection> thePoints(CSCSegmentCollection const* allCSCSegments,
                                                 edm::EventSetup const& iSetup,
                                                 bool debug,
                                                 double eyr);

private:
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeoToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeoToken_;
  edm::ESGetToken<CSCObjectMap, MuonGeometryRecord> cscMapToken_;
  int minBX;
  int maxBX;
};

#endif
