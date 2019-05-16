#ifndef MuonAlignment_H
#define MuonAlignment_H

/** \class MuonAlignment
 *
 *  DQM muon alignment analysis monitoring
 *
 *  \author J. Fernandez - Univ. Oviedo <Javier.Fernandez@cern.ch>
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <FWCore/Utilities/interface/InputTag.h>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "DQMServices/Core/interface/DQMStore.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
  class InputTag;
}  // namespace edm

class TH1F;

typedef std::vector<std::vector<int>> intDVector;
typedef std::vector<TrackingRecHit *> RecHitVector;

class MuonAlignment : public edm::EDAnalyzer {
public:
  /// Constructor
  MuonAlignment(const edm::ParameterSet &);

  /// Destructor
  ~MuonAlignment() override;

  /// Inizialize parameters for histo binning
  void beginJob() override;

  /// Get the analysis
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  /// Save the histos
  void endJob(void) override;

private:
  // ----------member data ---------------------------

  DQMStore *dbe;

  MonitorElement *hLocalPositionDT;
  MonitorElement *hLocalPositionRmsDT;
  MonitorElement *hLocalAngleDT;
  MonitorElement *hLocalAngleRmsDT;

  MonitorElement *hLocalXMeanDT;
  MonitorElement *hLocalXRmsDT;
  MonitorElement *hLocalYMeanDT;
  MonitorElement *hLocalYRmsDT;
  MonitorElement *hLocalPhiMeanDT;
  MonitorElement *hLocalPhiRmsDT;
  MonitorElement *hLocalThetaMeanDT;
  MonitorElement *hLocalThetaRmsDT;

  MonitorElement *hLocalPositionCSC;
  MonitorElement *hLocalPositionRmsCSC;
  MonitorElement *hLocalAngleCSC;
  MonitorElement *hLocalAngleRmsCSC;

  MonitorElement *hLocalXMeanCSC;
  MonitorElement *hLocalXRmsCSC;
  MonitorElement *hLocalYMeanCSC;
  MonitorElement *hLocalYRmsCSC;
  MonitorElement *hLocalPhiMeanCSC;
  MonitorElement *hLocalPhiRmsCSC;
  MonitorElement *hLocalThetaMeanCSC;
  MonitorElement *hLocalThetaRmsCSC;

  edm::ParameterSet parameters;

  // Switch for verbosity
  std::string metname;

  RecHitVector doMatching(const reco::Track &,
                          edm::Handle<DTRecSegment4DCollection> &,
                          edm::Handle<CSCSegmentCollection> &,
                          intDVector *,
                          intDVector *,
                          edm::ESHandle<GlobalTrackingGeometry> &);

  // Muon Track Label
  edm::EDGetTokenT<reco::TrackCollection> theMuonCollectionLabel;

  edm::EDGetTokenT<DTRecSegment4DCollection> theRecHits4DTagDT;

  edm::EDGetTokenT<CSCSegmentCollection> theRecHits4DTagCSC;

  std::string trackRefitterType;

  // residual histos residual range
  double resLocalXRangeStation1, resLocalXRangeStation2, resLocalXRangeStation3, resLocalXRangeStation4;
  double resLocalYRangeStation1, resLocalYRangeStation2, resLocalYRangeStation3, resLocalYRangeStation4;
  double resPhiRange, resThetaRange;

  // mean and rms histos ranges
  double meanPositionRange, rmsPositionRange, meanAngleRange, rmsAngleRange;

  // quality cuts for tracks and number of bins for residual histos
  unsigned int nbins, min1DTrackRecHitSize, min4DTrackSegmentSize;

  // flags to decide on subdetector and summary histograms
  bool doDT, doCSC, doSummary;

  // variables used
  //    Propagator * thePropagator;

  // Vector of chambers Residuals
  std::vector<MonitorElement *> unitsLocalX;
  std::vector<MonitorElement *> unitsLocalPhi;
  std::vector<MonitorElement *> unitsLocalTheta;
  std::vector<MonitorElement *> unitsLocalY;

  // Counters
  int numberOfTracks;
  int numberOfHits;

  // Top folder in root file
  std::string MEFolderName;
  std::stringstream topFolder;
};
#endif
