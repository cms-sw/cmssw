#ifndef  SIMHITTORPC_H
#define  SIMHITTORPC_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class SIMHITtoRPC {
public:
  explicit SIMHITtoRPC(std::vector<edm::Handle<edm::PSimHitContainer> > theSimHitContainers,const edm::EventSetup& iSetup, const edm::Event& iEvent, bool debug, int partid);
  ~SIMHITtoRPC();
  RPCRecHitCollection* thePoints(){return _ThePoints;}
   
private:
  RPCRecHitCollection* _ThePoints; 
  edm::OwnVector<RPCRecHit> RPCPointVector;
  bool inclsimhit;
  double MaxD;
};

#endif
