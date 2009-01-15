#ifndef RecoMuon_TrackingTools_MuonTimingExtractor_H
#define RecoMuon_TrackingTools_MuonTimingExtractor_H

/**\class MuonTimingExtractor
 *
 * Extracts timing information associated to a muon track
 *
*/
//
// Original Author:  Traczyk Piotr
//         Created:  Thu Oct 11 15:01:28 CEST 2007
// $Id: MuonTimingExtractor.h,v 1.2 2008/11/04 15:45:27 ptraczyk Exp $
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

class MuonTimingExtractor {

public:
  
  /// Constructor
  MuonTimingExtractor(const edm::ParameterSet&);
  
  /// Destructor
  ~MuonTimingExtractor();

 class TimeMeasurement
  {
   public:
     bool isLeft;
     bool isPhi;
     float posInLayer;
     float distIP;
     float timeCorr;
     int station;
     DetId driftCell;
  };

  reco::MuonTime fillTiming(edm::Event&, const edm::EventSetup&, reco::TrackRef muonTrack);

private:
  double fitT0(double &a, double &b, vector<double> xl, vector<double> yl, vector<double> xr, vector<double> yr );
  void rawFit(double &a, double &da, double &b, double &db, const vector<double> hitsx, const vector<double> hitsy);

  edm::InputTag DTSegmentTags_; 
  unsigned int theHitsMin;
  double thePruneCut;
  bool useSegmentT0;
  bool debug;
  
  MuonServiceProxy* theService;
  
  MuonSegmentMatcher *theMatcher;
};

#endif
