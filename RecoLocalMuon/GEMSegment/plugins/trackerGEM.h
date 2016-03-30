#ifndef GEMSegment_trackerGEM_h
#define GEMSegment_trackerGEM_h

/** \class trackerGEM 
 *
 * \author Jason Lee, based off ME0Muon
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DataFormats/GEMRecHit/interface/GEMSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

class FreeTrajectoryState;
class MagneticField;
class SteppingHelixPropagator;
class trackerGEM : public edm::EDProducer {
 public:
  /// Constructor
  explicit trackerGEM(const edm::ParameterSet&);
  /// Destructor
  ~trackerGEM();
  /// Produce the GEMSegment collection
  virtual void produce(edm::Event&, const edm::EventSetup&);

    
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endJob();

  reco::MuonChamberMatch* findGEMSegment(const reco::Track&, const GEMSegmentCollection&, int station, const SteppingHelixPropagator*);
    
  FreeTrajectoryState getFTS(const GlobalVector& , const GlobalVector& , 
			     int , const AlgebraicSymMatrix66& ,
			     const MagneticField* );

  FreeTrajectoryState getFTS(const GlobalVector& , const GlobalVector& , 
			     int , const AlgebraicSymMatrix55& ,
			     const MagneticField* );

  void getFromFTS(const FreeTrajectoryState& ,
		  GlobalVector& , GlobalVector& , 
		  int& , AlgebraicSymMatrix66& );

 private:


  edm::ESHandle<GEMGeometry> gemGeom;
  double maxPullXGE11_, maxDiffXGE11_, maxPullYGE11_, maxDiffYGE11_,
    maxPullXGE21_, maxDiffXGE21_, maxPullYGE21_, maxDiffYGE21_,
    maxDiffPhiDirection_;
  edm::EDGetTokenT<GEMSegmentCollection> gemSegmentsToken_;
  edm::EDGetTokenT<reco::TrackCollection> generalTracksToken_;

  float ntracks, nmatch, nmatch_ge11, nmatch_ge21;
};

#endif
