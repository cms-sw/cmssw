#ifndef  CSCSEGTORPC_H
#define  CSCSEGTORPC_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include <memory>

class CSCSegtoRPC {
public:
  CSCSegtoRPC(CSCSegmentCollection const* allCSCSegments, edm::EventSetup const& iSetup, bool debug, double eyr);
  ~CSCSegtoRPC();
  std::unique_ptr<RPCRecHitCollection> && thePoints(){ return std::move(_ThePoints); }
   
private:
  std::unique_ptr<RPCRecHitCollection> _ThePoints; 
  edm::OwnVector<RPCRecHit> RPCPointVector;
  bool inclcsc;
  double MaxD;
};

#endif
