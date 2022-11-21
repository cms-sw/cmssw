
#ifndef DTRunConditionVar_H
#define DTRunConditionVar_H

/** \class DTRunConditionVar
 *
 * Description:
 *
 *
 * \author : Paolo Bellan, Antonio Branca
 * $date   : 23/09/2011 15:42:04 CET $
 *
 * Modification:
 *
 */

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "CondFormats/DTObjects/interface/DTRecoConditions.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsVdriftRcd.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <vector>
#include <string>

class DTGeometry;
class DetLayer;
class DetId;

class DTRunConditionVar : public DQMEDAnalyzer {
public:
  //Constructor
  DTRunConditionVar(const edm::ParameterSet& pset);

  //Destructor
  ~DTRunConditionVar() override;

  //BookHistograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  //Operations
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  void bookChamberHistos(DQMStore::IBooker&, const DTChamberId& dtCh, std::string histoType, int, float, float);

  bool debug;
  int nMinHitsPhi;
  double maxAnglePhiSegm;

  edm::EDGetTokenT<DTRecSegment4DCollection> dt4DSegmentsToken_;

  edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeomToken_;
  const DTGeometry* dtGeom;

  edm::ESGetToken<DTMtime, DTMtimeRcd> mTimeToken_;
  const DTMtime* mTimeMap_;  // legacy DB object

  edm::ESGetToken<DTRecoConditions, DTRecoConditionsVdriftRcd> vDriftToken_;
  const DTRecoConditions* vDriftMap_;  // DB object in new format
  bool readLegacyVDriftDB;             // which one to use

  std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;

protected:
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
