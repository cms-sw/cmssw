#ifndef RecoMuon_MCSeedGenerator_MCMuonSeedGenerator_H
#define RecoMuon_MCSeedGenerator_MCMuonSeedGenerator_H

/** \class MCMuonSeedGenerator
 *  No description available.
 *
 *  $Date: 2008/06/27 15:01:51 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"


class MuonServiceProxy;
class TrajectorySeed;
class SimTrack;
class SimVertex;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MCMuonSeedGenerator : public edm::EDProducer {
 
public:

  enum SeedType{FromHits, FromTracks};

  
  /// Constructor
  MCMuonSeedGenerator(const edm::ParameterSet&);

  /// Destructor
  ~MCMuonSeedGenerator();

  // Operations

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  
  struct RadiusComparatorInOut{

    RadiusComparatorInOut(edm::ESHandle<GlobalTrackingGeometry> tg):theTG(tg){}
    
    bool operator()(const PSimHit *a,
		    const PSimHit *b) const{ 
      
      const GeomDet *geomDetA = theTG->idToDet(DetId(a->detUnitId()));
      const GeomDet *geomDetB = theTG->idToDet(DetId(b->detUnitId()));
      
      double distA = geomDetA->toGlobal(a->localPosition()).mag();
      double distB = geomDetB->toGlobal(b->localPosition()).mag();
      
      return distA < distB; 
    }
    
    edm::ESHandle<GlobalTrackingGeometry> theTG;
  };
  
private:
  TrajectorySeed* createSeedFromHit(const PSimHit*);
  TrajectorySeed* createSeedFromTrack(const SimTrack &simTrack, const SimVertex &simVertex, DetId detId);


  edm::InputTag theCSCSimHitLabel;
  edm::InputTag theDTSimHitLabel; 
  edm::InputTag theRPCSimHitLabel;
  edm::InputTag theSimTrackLabel;

  SeedType theSeedType;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;  
  
  double theErrorScale;
};

#endif

