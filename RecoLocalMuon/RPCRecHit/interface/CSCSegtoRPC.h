#ifndef  CSCSEGTORPC_H
#define  CSCSEGTORPC_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
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

class CSCStationIndex{
public:
  CSCStationIndex():_region(0),_station(0),_ring(0),_chamber(0){}
  CSCStationIndex(int region, int station, int ring, int chamber):
    _region(region),
    _station(station),
    _ring(ring),
    _chamber(chamber){}
  ~CSCStationIndex(){}
  int region() const {return _region;}
  int station() const {return _station;}
  int ring() const {return _ring;}
  int chamber() const {return _chamber;}
  bool operator<(const CSCStationIndex& cscind) const{
    if(cscind.region()!=this->region())
      return cscind.region()<this->region();
    else if(cscind.station()!=this->station())
      return cscind.station()<this->station();
    else if(cscind.ring()!=this->ring())
      return cscind.ring()<this->ring();
    else if(cscind.chamber()!=this->chamber())
      return cscind.chamber()<this->chamber();
    return false;
  }

private:
  int _region;
  int _station;
  int _ring;  
  int _chamber;
};

class ObjectMapCSC{
public:
  static ObjectMapCSC* GetInstance(const edm::EventSetup& iSetup);
  std::set<RPCDetId> GetRolls(CSCStationIndex cscstationindex){return mapInstance->rollstoreCSC[cscstationindex];}
//protected:
  std::map<CSCStationIndex,std::set<RPCDetId> > rollstoreCSC;
  ObjectMapCSC(const edm::EventSetup& iSetup);
private:
  static ObjectMapCSC* mapInstance;
}; 

#endif
