#ifndef DTCHAMBEREFFICIENCY_H
#define DTCHAMBEREFFICIENCY_H

/** \class DTChamberEfficiency
 *
 * Description:
 *
 * This class provides the histograms for the calculation of the
 * efficiency of muons reconstruction in the DTs. It is applicable
 * both in presence or absence of a magnetic field.
 * Histos are 2D Sector vs Chamber plots for each wheel
 *
 * \author : Mario Pelliccioni - INFN Torino <pellicci@cern.ch>
 * $date   : 05/12/2008 16:51:04 CET $
 *
 * Modification:
 *
 */

#include "DataFormats/Common/interface/Handle.h"


#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include <vector>

namespace reco {
  class TransientTrack;
}


class Chi2MeasurementEstimator;
class MuonServiceProxy;

class DQMStore;
class MonitorElement;
class FreeTrajectoryState;
class DetLayer;
class DetId;
class NavigationSchool;

class DTChamberEfficiency : public DQMEDAnalyzer
{

 public:
  //Constructor
  DTChamberEfficiency(const edm::ParameterSet& pset) ;

  //Destructor
  ~DTChamberEfficiency() ;

  //Operations
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
  void dqmBeginRun(const edm::Run& , const edm::EventSetup&);

 protected:
// Book the histograms
void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:

  //functions
  std::vector<const DetLayer*> compatibleLayers(const NavigationSchool& navigationSchool, const DetLayer *initialLayer,
						const FreeTrajectoryState& fts, PropagationDirection propDir);

  MeasurementContainer segQualityCut(const MeasurementContainer& seg_list) const;
  bool chamberSelection(const DetId& idDetLay, reco::TransientTrack& trans_track) const;
  inline edm::ESHandle<Propagator> propagator() const;

  //data members
  bool debug;

  edm::InputTag theTracksLabel_;
  edm::EDGetTokenT<reco::TrackCollection> theTracksToken_;

  edm::InputTag labelRPCRecHits;
  edm::InputTag thedt4DSegments;
  edm::InputTag thecscSegments;

  double theMaxChi2;
  double theNSigma;
  int theMinNrec;

  std::string theNavigationType;

  edm::ESHandle<DTGeometry> dtGeom;

  MuonServiceProxy* theService;
  MuonDetLayerMeasurements* theMeasurementExtractor;
  Chi2MeasurementEstimator* theEstimator;

  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;

  std::vector<std::vector<MonitorElement*> > histosPerW;

 protected:

};

#endif // DTANALYZER_H

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
