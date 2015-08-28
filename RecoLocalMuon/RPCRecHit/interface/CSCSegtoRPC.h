#ifndef  CSCSEGTORPC_H
#define  CSCSEGTORPC_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

class CSCSegtoRPC {
public:
  explicit CSCSegtoRPC(edm::Handle<CSCSegmentCollection> allCSCSegments,const edm::EventSetup& iSetup, const edm::Event& iEvent, bool debug, double eyr);
  ~CSCSegtoRPC();
  RPCRecHitCollection* thePoints(){return _ThePoints;}
   
private:
  RPCRecHitCollection* _ThePoints; 
  edm::OwnVector<RPCRecHit> RPCPointVector;
  bool inclcsc;
  double MaxD;
};

#endif
