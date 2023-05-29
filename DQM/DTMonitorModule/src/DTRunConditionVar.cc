
/******* \class DTRunConditionVar *******
 *
 * Description:
 *
 *  detailed description
 *
 * \author : Paolo Bellan, Antonio Branca
 * $date   : 23/09/2011 15:42:04 CET $
 *
 * Modification:
 *
 *********************************/

#include "DTRunConditionVar.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "TMath.h"
#include <cmath>

using namespace std;
using namespace edm;

DTRunConditionVar::DTRunConditionVar(const ParameterSet& pSet)
    :  // Get the debug parameter for verbose output
      debug(pSet.getUntrackedParameter<bool>("debug", false)),
      nMinHitsPhi(pSet.getUntrackedParameter<int>("nMinHitsPhi")),
      maxAnglePhiSegm(pSet.getUntrackedParameter<double>("maxAnglePhiSegm")),
      dt4DSegmentsToken_(consumes<DTRecSegment4DCollection>(pSet.getUntrackedParameter<InputTag>("recoSegments"))),
      muonGeomToken_(esConsumes<edm::Transition::BeginRun>()),
      readLegacyVDriftDB(pSet.getParameter<bool>("readLegacyVDriftDB")) {
  if (readLegacyVDriftDB) {
    mTimeToken_ = esConsumes();
  } else {
    vDriftToken_ = esConsumes();
  }
}

DTRunConditionVar::~DTRunConditionVar() {
  LogTrace("DTDQM|DTMonitorModule|DTRunConditionVar") << "DTRunConditionVar: destructor called";

  // free memory
}

void DTRunConditionVar::bookHistograms(DQMStore::IBooker& ibooker,
                                       edm::Run const& iRun,
                                       edm::EventSetup const& /* iSetup */) {
  LogTrace("DTDQM|DTMonitorModule|DTRunConditionVar") << "DTRunConditionVar: bookHistograms";

  for (int wheel = -2; wheel <= 2; wheel++) {
    for (int sec = 1; sec <= 14; sec++) {
      for (int stat = 1; stat <= 4; stat++) {
        bookChamberHistos(ibooker, DTChamberId(wheel, stat, sec), "VDrift_FromSegm", 100, 0.0043, 0.0065);
        bookChamberHistos(ibooker, DTChamberId(wheel, stat, sec), "T0_FromSegm", 100, -25., 25.);
      }
    }
  }

  return;
}

void DTRunConditionVar::dqmBeginRun(const Run& run, const EventSetup& setup) {
  // Get the DT Geometry
  dtGeom = &setup.getData(muonGeomToken_);
  return;
}

void DTRunConditionVar::analyze(const Event& event, const EventSetup& eventSetup) {
  LogTrace("DTDQM|DTMonitorModule|DTRunConditionVar")
      << "--- [DTRunConditionVar] Event analysed #Run: " << event.id().run() << " #Event: " << event.id().event()
      << endl;

  // Get the map of vdrift from the setup
  if (readLegacyVDriftDB) {
    mTimeMap_ = &eventSetup.getData(mTimeToken_);
    vDriftMap_ = nullptr;
  } else {
    vDriftMap_ = &eventSetup.getData(vDriftToken_);
    mTimeMap_ = nullptr;
    // Consistency check: no parametrization is implemented for the time being
    int version = vDriftMap_->version();
    if (version != 1) {
      throw cms::Exception("Configuration") << "only version 1 is presently supported for VDriftDB";
    }
  }

  // Get the segment collection from the event
  Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByToken(dt4DSegmentsToken_, all4DSegments);

  // Loop over the segments
  for (DTRecSegment4DCollection::const_iterator segment = all4DSegments->begin(); segment != all4DSegments->end();
       ++segment) {
    // Get the chamber from the setup
    DTChamberId DTid = (DTChamberId)segment->chamberId();
    uint32_t indexCh = DTid.rawId();

    // Fill v-drift values
    if ((*segment).hasPhi()) {
      int nHitsPhi = (*segment).phiSegment()->degreesOfFreedom() + 2;
      double xdir = (*segment).phiSegment()->localDirection().x();
      double zdir = (*segment).phiSegment()->localDirection().z();

      double anglePhiSegm = fabs(atan(xdir / zdir)) * 180. / TMath::Pi();

      if (nHitsPhi >= nMinHitsPhi && anglePhiSegm <= maxAnglePhiSegm) {
        double segmentVDrift = segment->phiSegment()->vDrift();

        DTSuperLayerId indexSLPhi1(DTid, 1);
        DTSuperLayerId indexSLPhi2(DTid, 3);

        float vDriftPhi1(0.), vDriftPhi2(0.);
        float ResPhi1(0.), ResPhi2(0.);
        if (readLegacyVDriftDB) {  // Legacy format
          int status1 = mTimeMap_->get(indexSLPhi1, vDriftPhi1, ResPhi1, DTVelocityUnits::cm_per_ns);
          int status2 = mTimeMap_->get(indexSLPhi2, vDriftPhi2, ResPhi2, DTVelocityUnits::cm_per_ns);

          if (status1 != 0 || status2 != 0) {
            DTSuperLayerId sl = (status1 != 0) ? indexSLPhi1 : indexSLPhi2;
            throw cms::Exception("DTRunConditionVarClient") << "Could not find vDrift entry in DB for" << sl << endl;
          }
        } else {
          vDriftPhi1 = vDriftMap_->get(DTWireId(indexSLPhi1.rawId()));
          vDriftPhi2 = vDriftMap_->get(DTWireId(indexSLPhi2.rawId()));
        }

        float vDriftMed = (vDriftPhi1 + vDriftPhi2) / 2.;

        segmentVDrift = vDriftMed * (1. - segmentVDrift);

        double segmentT0 = segment->phiSegment()->t0();

        if (segment->phiSegment()->ist0Valid())
          (chamberHistos[indexCh])["T0_FromSegm"]->Fill(segmentT0);
        if (segmentVDrift != vDriftMed)
          (chamberHistos[indexCh])["VDrift_FromSegm"]->Fill(segmentVDrift);
      }
    }

  }  //end loop on segment

}  //end analyze

void DTRunConditionVar::bookChamberHistos(
    DQMStore::IBooker& ibooker, const DTChamberId& dtCh, string histoType, int nbins, float min, float max) {
  int wh = dtCh.wheel();
  int sc = dtCh.sector();
  int st = dtCh.station();
  stringstream wheel;
  wheel << wh;
  stringstream station;
  station << st;
  stringstream sector;
  sector << sc;

  string bookingFolder = "DT/02-Segments/Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" + station.str();
  string histoTag = "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  ibooker.setCurrentFolder(bookingFolder);

  LogTrace("DTDQM|DTMonitorModule|DTRunConditionVar")
      << "[DTRunConditionVar]: booking histos in " << bookingFolder << endl;

  string histoName = histoType + histoTag;
  const string& histoLabel = histoType;

  (chamberHistos[dtCh.rawId()])[histoType] = ibooker.book1D(histoName, histoLabel, nbins, min, max);
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
