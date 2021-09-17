// -*- C++ -*-
//
// Package:    CalibTracker/SiStripChannelGain
// Class:      SiStripGainsPCLWorker
//
/**\class SiStripGainsPCLWorker SiStripGainsPCLWorker.cc 
   Description: Fill DQM histograms with SiStrip Charge normalized to path length
 
*/
//
//  Original Author: L. Quertermont (calibration algorithm)
//  Contributors:    M. Verzetti    (data access)
//                   A. Di Mattia   (PCL multi stream processing and monitoring)
//                   M. Delcourt    (monitoring)
//                   M. Musich      (migration to thread-safe DQMStore access)
//                   P. David       (merge ShallowGainCalibration with SiStripGainsPCLWorker)
//
//  Created:  Wed, 12 Apr 2017 14:46:48 GMT
//

// CMSSW includes
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

/// user includes
#include "CalibTracker/SiStripChannelGain/interface/APVGainStruct.h"
#include "CalibTracker/SiStripChannelGain/interface/APVGainHelpers.h"

// System includes
#include <unordered_map>

//
// class declaration
//

class SiStripGainsPCLWorker : public DQMGlobalEDAnalyzer<APVGain::APVGainHistograms> {
public:
  explicit SiStripGainsPCLWorker(const edm::ParameterSet &);

  void bookHistograms(DQMStore::IBooker &,
                      edm::Run const &,
                      edm::EventSetup const &,
                      APVGain::APVGainHistograms &) const override;
  void dqmAnalyze(edm::Event const &, edm::EventSetup const &, APVGain::APVGainHistograms const &) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &, APVGain::APVGainHistograms &) const override;
  void checkBookAPVColls(const TrackerGeometry *bareTkGeomPtr, APVGain::APVGainHistograms &histograms) const;

  std::vector<std::string> dqm_tag_;

  int statCollectionFromMode(const char *tag) const;

  double MinTrackMomentum;
  double MaxTrackMomentum;
  double MinTrackEta;
  double MaxTrackEta;
  unsigned int MaxNrStrips;
  unsigned int MinTrackHits;
  double MaxTrackChiOverNdf;
  int MaxTrackingIteration;
  bool AllowSaturation;
  bool FirstSetOfConstants;
  bool Validation;
  bool OldGainRemoving;
  bool useCalibration;
  bool doChargeMonitorPerPlane; /*!< Charge monitor per detector plane */

  std::string m_DQMdir;                  /*!< DQM folder hosting the charge statistics and the monitor plots */
  std::string m_calibrationMode;         /*!< Type of statistics for the calibration */
  std::vector<std::string> VChargeHisto; /*!< Charge monitor plots to be output */

  edm::EDGetTokenT<edm::View<reco::Track>> m_tracks_token;
  edm::EDGetTokenT<TrajTrackAssociationCollection> m_association_token;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomTokenBR_, tkGeomToken_;
  edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;

  // maps histograms index to topology
  std::map<unsigned int, APVloc> theTopologyMap;
};

inline int SiStripGainsPCLWorker::statCollectionFromMode(const char *tag) const {
  std::vector<std::string>::const_iterator it = dqm_tag_.begin();
  while (it != dqm_tag_.end()) {
    if (*it == std::string(tag))
      return it - dqm_tag_.begin();
    it++;
  }

  if (std::string(tag).empty())
    return 0;  // return StdBunch calibration mode for backward compatibility

  return None;
}
