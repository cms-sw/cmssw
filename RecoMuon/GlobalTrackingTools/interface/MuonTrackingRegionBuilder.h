#ifndef RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H
#define RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H

/** \class MuonTrackingRegionBuilder
 *  Base class for the Muon reco TrackingRegion Builder
 *
 *  $Date: 2007/05/09 19:28:20 $
 *  $Revision: 1.1 $
 *  \author A. Everett - Purdue University
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

class MuonServiceProxy;
class RectangularEtaPhiTrackingRegion;

//namespace reco {class Track; class TrackRef;}

class MuonTrackingRegionBuilder {
  
 public:
  
  /// constructor
  MuonTrackingRegionBuilder(const edm::ParameterSet&, const MuonServiceProxy*);
  
  /// destructor
  virtual ~MuonTrackingRegionBuilder() {}
  
  RectangularEtaPhiTrackingRegion* region(const reco::TrackRef&) const;

  RectangularEtaPhiTrackingRegion* region(const reco::Track&) const;
  
 private:
  edm::ParameterSet theRegionPSet;
  const MuonServiceProxy * theService;

  bool theMakeTkSeedFlag;

  GlobalPoint theVertexPos;
  GlobalError theVertexErr;
  

};
#endif
