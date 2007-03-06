#ifndef RecoMuon_MCSeedGenerator_MCSeedGenerator_H
#define RecoMuon_MCSeedGenerator_MCSeedGenerator_H

/** \class MCSeedGenerator
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
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

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MCSeedGenerator : public edm::EDProducer {
 
public:
  
  /// Constructor
  MCSeedGenerator(const edm::ParameterSet&);

  /// Destructor
  ~MCSeedGenerator();

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
  TrajectorySeed* createSeed(const PSimHit*);

  edm::InputTag theCSCSimHitLabel;
  edm::InputTag theDTSimHitLabel; 
  edm::InputTag theRPCSimHitLabel;
  edm::InputTag theSimTrackLabel;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;  

  double theErrorScale;
};

#endif

