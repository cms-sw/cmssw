#ifndef  DTSEGTORPC_H
#define  DTSEGTORPC_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include <memory>

class DTSegtoRPC {
public:
  DTSegtoRPC(DTRecSegment4DCollection const* all4DSegments, edm::EventSetup const& iSetup, bool debug, double eyr);
  ~DTSegtoRPC();
  std::unique_ptr<RPCRecHitCollection> && thePoints(){ return std::move(_ThePoints); }
   
private:
  std::unique_ptr<RPCRecHitCollection> _ThePoints; 
  edm::OwnVector<RPCRecHit> RPCPointVector;
  bool incldt;
  bool incldtMB4;
  double MinCosAng;
  double MaxD;
  double MaxDrb4;
  double MaxDistanceBetweenSegments;
  std::vector<uint32_t> extrapolatedRolls;
};

#endif
