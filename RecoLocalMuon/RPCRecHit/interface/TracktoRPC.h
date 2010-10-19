#ifndef  TRACKTORPC_H
#define  TRACKTORPC_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/Common/interface/Ref.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformerForCosmicMuons.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformerBase.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
//#include "RecoLocalMuon/RPCRecHit/interface/DTSegtoRPC.h"
//#include "RecoLocalMuon/RPCRecHit/interface/CSCSegtoRPC.h"

using reco::MuonCollection;
using reco::TrackCollection;
typedef std::vector<Trajectory> Trajectories;

class TracktoRPC {
public:


  explicit TracktoRPC(edm::Handle<reco::TrackCollection> alltracks,const edm::EventSetup& iSetup, const edm::Event& iEvent,bool debug, const edm::ParameterSet& iConfig,edm::InputTag& tracklabel);

  ~TracktoRPC();
  RPCRecHitCollection* thePoints(){return _ThePoints;}
  bool ValidRPCSurface(RPCDetId rpcid, LocalPoint LocalP, const edm::EventSetup& iSetup);

private:
  RPCRecHitCollection* _ThePoints;
  edm::OwnVector<RPCRecHit> RPCPointVector;
  double MaxD;

 TrackTransformerBase *theTrackTransformer;
 edm::ESHandle<Propagator> thePropagator;
};

class DTStationIndex2{
public: 
  DTStationIndex2():_region(0),_wheel(0),_sector(0),_station(0){}
  DTStationIndex2(int region, int wheel, int sector, int station) : 
    _region(region),
    _wheel(wheel),
    _sector(sector),
    _station(station){}
  ~DTStationIndex2(){}
  int region() const {return _region;}
  int wheel() const {return _wheel;}
  int sector() const {return _sector;}
  int station() const {return _station;}
  bool operator<(const DTStationIndex2& dtind) const{
    if(dtind.region()!=this->region())
      return dtind.region()<this->region();
    else if(dtind.wheel()!=this->wheel())
      return dtind.wheel()<this->wheel();
    else if(dtind.sector()!=this->sector())
      return dtind.sector()<this->sector();
    else if(dtind.station()!=this->station())
      return dtind.station()<this->station();
    return false;
  }

private:
  int _region;
  int _wheel;
  int _sector;
  int _station; 
};

class ObjectMap2{
public:
  static ObjectMap2* GetInstance(const edm::EventSetup& iSetup);
  std::set<RPCDetId> GetRolls(DTStationIndex2 dtstationindex){return mapInstance->rollstoreDT[dtstationindex];}
//protected:
  std::map<DTStationIndex2,std::set<RPCDetId> > rollstoreDT;
  ObjectMap2(const edm::EventSetup& iSetup);
private:
  static ObjectMap2* mapInstance;
}; 
class CSCStationIndex2{
public:
  CSCStationIndex2():_region(0),_station(0),_ring(0),_chamber(0){}
  CSCStationIndex2(int region, int station, int ring, int chamber):
    _region(region),
    _station(station),
    _ring(ring),
    _chamber(chamber){}
  ~CSCStationIndex2(){}
  int region() const {return _region;}
  int station() const {return _station;}
  int ring() const {return _ring;}
  int chamber() const {return _chamber;}
  bool operator<(const CSCStationIndex2& cscind) const{
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

class ObjectMap2CSC{
public:
  static ObjectMap2CSC* GetInstance(const edm::EventSetup& iSetup);
  std::set<RPCDetId> GetRolls(CSCStationIndex2 cscstationindex){return mapInstance->rollstoreCSC[cscstationindex];}
//protected:
  std::map<CSCStationIndex2,std::set<RPCDetId> > rollstoreCSC;
  ObjectMap2CSC(const edm::EventSetup& iSetup);
private:
  static ObjectMap2CSC* mapInstance;
}; 

#endif
