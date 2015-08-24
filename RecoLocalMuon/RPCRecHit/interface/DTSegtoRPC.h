#ifndef  DTSEGTORPC_H
#define  DTSEGTORPC_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"


class DTSegtoRPC {
public:
  explicit DTSegtoRPC(edm::Handle<DTRecSegment4DCollection> all4DSegments,const edm::EventSetup& iSetup, const edm::Event& iEvent,bool debug, double eyr);
  ~DTSegtoRPC();
  RPCRecHitCollection* thePoints(){return _ThePoints;}
   
private:
  RPCRecHitCollection* _ThePoints; 
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
