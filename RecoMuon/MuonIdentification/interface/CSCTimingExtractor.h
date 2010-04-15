#ifndef MuonIdentification_CSCTimingExtractor_H
#define MuonIdentification_CSCTimingExtractor_H

/**\class CSCTimingExtractor
 *
 * Extracts timing information associated to a muon track
 *
*/
//
// Original Author:  Traczyk Piotr
//         Created:  Thu Oct 11 15:01:28 CEST 2007
// $Id: CSCTimingExtractor.h,v 1.1 2009/03/27 02:26:41 ptraczyk Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/Common/interface/Ref.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonSegmentMatcher.h"
#include "RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include <vector>

namespace edm {
  class ParameterSet;
  class EventSetup;
  class InputTag;
}

class MuonServiceProxy;

using namespace std;

class CSCTimingExtractor {

public:
  
  /// Constructor
  CSCTimingExtractor(const edm::ParameterSet&);
  
  /// Destructor
  ~CSCTimingExtractor();

 class TimeMeasurement
  {
   public:
     float distIP;
     float timeCorr;
     int station;
  };

  void fillTiming(TimeMeasurementSequence &tmSequence, reco::TrackRef muonTrack, edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  edm::InputTag CSCSegmentTags_; 
  unsigned int theHitsMin_;
  double thePruneCut_;
  double theTimeOffset_;
  bool debug;
  
  MuonServiceProxy* theService;
  
  MuonSegmentMatcher *theMatcher;
};

#endif
