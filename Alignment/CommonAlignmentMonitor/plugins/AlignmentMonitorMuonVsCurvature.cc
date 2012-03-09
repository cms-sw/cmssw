// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorMuonVsCurvature
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Fri Feb 19 21:45:02 CET 2010
//

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsPositionFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsAngleFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsTwoBin.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <sstream>
#include "TProfile.h"
#include "TH2F.h"
#include "TH1F.h"

// user include files

// 
// class definition
// 

class AlignmentMonitorMuonVsCurvature: public AlignmentMonitorBase {
   public:
      AlignmentMonitorMuonVsCurvature(const edm::ParameterSet& cfg);
      ~AlignmentMonitorMuonVsCurvature() {};

      void book();
      void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
      void afterAlignment(const edm::EventSetup &iSetup);

   private:
      double m_minTrackPt;
      int m_minTrackerHits;
      double m_maxTrackerRedChi2;
      bool m_allowTIDTEC;
      double m_maxDxy;
      int m_minDT13Hits;
      int m_minDT2Hits;
      int m_minCSCHits;
      int m_layer;
      std::string m_propagator;

      enum {
	 kWheelMinus2 = 0,
	 kWheelMinus1,
	 kWheelZero,
	 kWheelPlus1,
	 kWheelPlus2,
	 kWheelMEm11,
	 kWheelMEm12,
	 kWheelMEm13,
	 kWheelMEm14,
	 kWheelMEp11,
	 kWheelMEp12,
	 kWheelMEp13,
	 kWheelMEp14,
	 kNumWheels
      };

      enum {
	 kEndcapMEm11 = 0,
	 kEndcapMEm12,
	 kEndcapMEm13,
	 kEndcapMEm14,
	 kEndcapMEp11,
	 kEndcapMEp12,
	 kEndcapMEp13,
	 kEndcapMEp14,
	 kNumEndcap
      };

      enum {
	 kDeltaX = 0,
	 kDeltaDxDz,
	 kPtErr,
	 kCurvErr,
	 kNumComponents
      };

      TH2F *th2f_wheelsector[kNumWheels][12][kNumComponents];
      TProfile *tprofile_wheelsector[kNumWheels][12][kNumComponents];

      TH2F *th2f_evens[kNumEndcap][kNumComponents];
      TH2F *th2f_odds[kNumEndcap][kNumComponents];
      TProfile *tprofile_evens[kNumEndcap][kNumComponents];
      TProfile *tprofile_odds[kNumEndcap][kNumComponents];

      TH2F *th2f_endcap[kNumEndcap][36][kNumComponents];
      TProfile *tprofile_endcap[kNumEndcap][36][kNumComponents];

      TH1F *th1f_trackerRedChi2;
      TH1F *th1f_trackerRedChi2Diff;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// member functions
//

AlignmentMonitorMuonVsCurvature::AlignmentMonitorMuonVsCurvature(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorMuonVsCurvature")
   , m_minTrackPt(cfg.getParameter<double>("minTrackPt"))
   , m_minTrackerHits(cfg.getParameter<int>("minTrackerHits"))
   , m_maxTrackerRedChi2(cfg.getParameter<double>("maxTrackerRedChi2"))
   , m_allowTIDTEC(cfg.getParameter<bool>("allowTIDTEC"))
   , m_maxDxy(cfg.getParameter<double>("maxDxy"))
   , m_minDT13Hits(cfg.getParameter<int>("minDT13Hits"))
   , m_minDT2Hits(cfg.getParameter<int>("minDT2Hits"))
   , m_minCSCHits(cfg.getParameter<int>("minCSCHits"))
   , m_layer(cfg.getParameter<int>("layer"))
   , m_propagator(cfg.getParameter<std::string>("propagator"))
{}

void AlignmentMonitorMuonVsCurvature::book() {
   for (int wheel = 0;  wheel < kNumWheels;  wheel++) {
      std::stringstream wheelname;
      if (wheel == kWheelMinus2) wheelname << "wheelm2_";
      else if (wheel == kWheelMinus1) wheelname << "wheelm1_";
      else if (wheel == kWheelZero) wheelname << "wheelz_";
      else if (wheel == kWheelPlus1) wheelname << "wheelp1_";
      else if (wheel == kWheelPlus2) wheelname << "wheelp2_";
      else if (wheel == kWheelMEm11) wheelname << "wheelmem11_";
      else if (wheel == kWheelMEm12) wheelname << "wheelmem12_";
      else if (wheel == kWheelMEm13) wheelname << "wheelmem13_";
      else if (wheel == kWheelMEm14) wheelname << "wheelmem14_";
      else if (wheel == kWheelMEp11) wheelname << "wheelmep11_";
      else if (wheel == kWheelMEp12) wheelname << "wheelmep12_";
      else if (wheel == kWheelMEp13) wheelname << "wheelmep13_";
      else if (wheel == kWheelMEp14) wheelname << "wheelmep14_";

      for (int sector = 0;  sector < 12;  sector++) {
	 std::stringstream sectorname;
	 if (sector == 0) sectorname << "sector01_";
	 else if (sector == 1) sectorname << "sector02_";
	 else if (sector == 2) sectorname << "sector03_";
	 else if (sector == 3) sectorname << "sector04_";
	 else if (sector == 4) sectorname << "sector05_";
	 else if (sector == 5) sectorname << "sector06_";
	 else if (sector == 6) sectorname << "sector07_";
	 else if (sector == 7) sectorname << "sector08_";
	 else if (sector == 8) sectorname << "sector09_";
	 else if (sector == 9) sectorname << "sector10_";
	 else if (sector == 10) sectorname << "sector11_";
	 else if (sector == 11) sectorname << "sector12_";

	 for (int component = 0;  component < kNumComponents;  component++) {
	    std::stringstream th2f_name, tprofile_name;
	    th2f_name << "th2f_" << wheelname.str() << sectorname.str();
	    tprofile_name << "tprofile_" << wheelname.str() << sectorname.str();

	    double minmax = 15.;
	    if (component == kDeltaX) {
	       th2f_name << "deltax";
	       tprofile_name << "deltax";
	       minmax = 15.;
	    }
	    else if (component == kDeltaDxDz) {
	       th2f_name << "deltadxdz";
	       tprofile_name << "deltadxdz";
	       minmax = 15.;
	    }
	    else if (component == kPtErr) {
	       th2f_name << "pterr";
	       tprofile_name << "pterr";
	       minmax = 15.;
	    }
	    else if (component == kCurvErr) {
	       th2f_name << "curverr";
	       tprofile_name << "curverr";
	       minmax = 0.0015;
	    }

	    if (component == kPtErr) {
	       th2f_wheelsector[wheel][sector][component] = book2D("/iterN/", th2f_name.str().c_str(), "", 25, -200., 200., 30, -minmax, minmax);
	       tprofile_wheelsector[wheel][sector][component] = bookProfile("/iterN/", tprofile_name.str().c_str(), "", 25, -200., 200.);
	    }
	    else {
	       th2f_wheelsector[wheel][sector][component] = book2D("/iterN/", th2f_name.str().c_str(), "", 25, -0.05, 0.05, 30, -minmax, minmax);
	       tprofile_wheelsector[wheel][sector][component] = bookProfile("/iterN/", tprofile_name.str().c_str(), "", 25, -0.05, 0.05);
	    }
	 }
      }
   }

   for (int endcap = 0;  endcap < kNumEndcap;  endcap++) {
      std::stringstream endcapname;
      if (endcap == kEndcapMEm11) endcapname << "endcapmem11_";
      else if (endcap == kEndcapMEm12) endcapname << "endcapmem12_";
      else if (endcap == kEndcapMEm13) endcapname << "endcapmem13_";
      else if (endcap == kEndcapMEm14) endcapname << "endcapmem14_";
      else if (endcap == kEndcapMEp11) endcapname << "endcapmep11_";
      else if (endcap == kEndcapMEp12) endcapname << "endcapmep12_";
      else if (endcap == kEndcapMEp13) endcapname << "endcapmep13_";
      else if (endcap == kEndcapMEp14) endcapname << "endcapmep14_";

      for (int component = 0;  component < kNumComponents;  component++) {
	 std::stringstream componentname;
	 double minmax = 15.;
	 if (component == kDeltaX) {
	    componentname << "deltax";
	    minmax = 15.;
	 }
	 else if (component == kDeltaDxDz) {
	    componentname << "deltadxdz";
	    minmax = 15.;
	 }
	 else if (component == kPtErr) {
	    componentname << "pterr";
	    minmax = 15.;
	 }
	 else if (component == kCurvErr) {
	    componentname << "curverr";
	    minmax = 0.0015;
	 }

	 std::stringstream th2f_evens_name, th2f_odds_name, tprofile_evens_name, tprofile_odds_name;
	 th2f_evens_name << "th2f_" << endcapname.str() << "evens_" << componentname.str();
	 th2f_odds_name << "th2f_" << endcapname.str() << "odds_" << componentname.str();
	 tprofile_evens_name << "tprofile_" << endcapname.str() << "evens_" << componentname.str();
	 tprofile_odds_name << "tprofile_" << endcapname.str() << "odds_" << componentname.str();
	 
	 if (component == kPtErr) {
	    th2f_evens[endcap][component] = book2D("/iterN/", th2f_evens_name.str().c_str(), "", 25, -200., 200., 30, -minmax, minmax);
	    th2f_odds[endcap][component] = book2D("/iterN/", th2f_odds_name.str().c_str(), "", 25, -200., 200., 30, -minmax, minmax);
	    tprofile_evens[endcap][component] = bookProfile("/iterN/", tprofile_evens_name.str().c_str(), "", 25, -200., 200.);
	    tprofile_odds[endcap][component] = bookProfile("/iterN/", tprofile_odds_name.str().c_str(), "", 25, -200., 200.);
	 }
	 else {
	    th2f_evens[endcap][component] = book2D("/iterN/", th2f_evens_name.str().c_str(), "", 25, -0.05, 0.05, 30, -minmax, minmax);
	    th2f_odds[endcap][component] = book2D("/iterN/", th2f_odds_name.str().c_str(), "", 25, -0.05, 0.05, 30, -minmax, minmax);
	    tprofile_evens[endcap][component] = bookProfile("/iterN/", tprofile_evens_name.str().c_str(), "", 25, -0.05, 0.05);
	    tprofile_odds[endcap][component] = bookProfile("/iterN/", tprofile_odds_name.str().c_str(), "", 25, -0.05, 0.05);
	 }

	 for (int chamber = 0;  chamber < 36;  chamber++) {
	    std::stringstream chambername;
	    if (chamber == 0) chambername << "chamber01_";
	    else if (chamber == 1) chambername << "chamber02_";
	    else if (chamber == 2) chambername << "chamber03_";
	    else if (chamber == 3) chambername << "chamber04_";
	    else if (chamber == 4) chambername << "chamber05_";
	    else if (chamber == 5) chambername << "chamber06_";
	    else if (chamber == 6) chambername << "chamber07_";
	    else if (chamber == 7) chambername << "chamber08_";
	    else if (chamber == 8) chambername << "chamber09_";
	    else if (chamber == 9) chambername << "chamber10_";
	    else if (chamber == 10) chambername << "chamber11_";
	    else if (chamber == 11) chambername << "chamber12_";
	    else if (chamber == 12) chambername << "chamber13_";
	    else if (chamber == 13) chambername << "chamber14_";
	    else if (chamber == 14) chambername << "chamber15_";
	    else if (chamber == 15) chambername << "chamber16_";
	    else if (chamber == 16) chambername << "chamber17_";
	    else if (chamber == 17) chambername << "chamber18_";
	    else if (chamber == 18) chambername << "chamber19_";
	    else if (chamber == 19) chambername << "chamber20_";
	    else if (chamber == 20) chambername << "chamber21_";
	    else if (chamber == 21) chambername << "chamber22_";
	    else if (chamber == 22) chambername << "chamber23_";
	    else if (chamber == 23) chambername << "chamber24_";
	    else if (chamber == 24) chambername << "chamber25_";
	    else if (chamber == 25) chambername << "chamber26_";
	    else if (chamber == 26) chambername << "chamber27_";
	    else if (chamber == 27) chambername << "chamber28_";
	    else if (chamber == 28) chambername << "chamber29_";
	    else if (chamber == 29) chambername << "chamber30_";
	    else if (chamber == 30) chambername << "chamber31_";
	    else if (chamber == 31) chambername << "chamber32_";
	    else if (chamber == 32) chambername << "chamber33_";
	    else if (chamber == 33) chambername << "chamber34_";
	    else if (chamber == 34) chambername << "chamber35_";
	    else if (chamber == 35) chambername << "chamber36_";

	    std::stringstream th2f_name, tprofile_name;
	    th2f_name << "th2f_" << endcapname.str() << chambername.str() << componentname.str();
	    tprofile_name << "tprofile_" << endcapname.str() << chambername.str() << componentname.str();

	    if (component == kPtErr) {
	       th2f_endcap[endcap][chamber][component] = book2D("/iterN/", th2f_name.str().c_str(), "", 25, -200., 200., 30, -minmax, minmax);
	       tprofile_endcap[endcap][chamber][component] = bookProfile("/iterN/", tprofile_name.str().c_str(), "", 25, -200., 200.);
	    }
	    else {
	       th2f_endcap[endcap][chamber][component] = book2D("/iterN/", th2f_name.str().c_str(), "", 25, -0.05, 0.05, 30, -minmax, minmax);
	       tprofile_endcap[endcap][chamber][component] = bookProfile("/iterN/", tprofile_name.str().c_str(), "", 25, -0.05, 0.05);
	    }

	 }
      }
   }

   th1f_trackerRedChi2 = book1D("/iterN/", "trackerRedChi2", "Refit tracker reduced chi^2", 100, 0., 30.);
   th1f_trackerRedChi2Diff = book1D("/iterN/", "trackerRedChi2Diff", "Fit-minus-refit tracker reduced chi^2", 100, -5., 5.);
}

void AlignmentMonitorMuonVsCurvature::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& trajtracks) {
   edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
   iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get(m_propagator, propagator);

   edm::ESHandle<MagneticField> magneticField;
   iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

   for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack) {
      const Trajectory* traj = (*trajtrack).first;
      const reco::Track* track = (*trajtrack).second;

      if (track->pt() > m_minTrackPt  &&  fabs(track->dxy()) < m_maxDxy) {
	 MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, pNavigator(), 1000.);

	 if (muonResidualsFromTrack.trackerNumHits() >= m_minTrackerHits  &&  muonResidualsFromTrack.trackerRedChi2() < m_maxTrackerRedChi2  &&  (m_allowTIDTEC  ||  !muonResidualsFromTrack.contains_TIDTEC())) {
	    std::vector<DetId> chamberIds = muonResidualsFromTrack.chamberIds();

	    double qoverpt = track->charge() / track->pt();
	    double px = track->px();
	    double py = track->py();
	    double pz = track->pz();

	    th1f_trackerRedChi2->Fill(muonResidualsFromTrack.trackerRedChi2());
	    th1f_trackerRedChi2Diff->Fill(track->normalizedChi2() - muonResidualsFromTrack.trackerRedChi2());

	    for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId) {
	       if (chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::DT) {
		  DTChamberId dtid(chamberId->rawId());
		  if (dtid.station() == 1) {
	      
		     MuonChamberResidual *dt13 = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kDT13);

		     if (dt13 != NULL  &&  dt13->numHits() >= m_minDT13Hits) {
			int wheel = -1;
			if (dtid.wheel() == -2) wheel = kWheelMinus2;
			else if (dtid.wheel() == -1) wheel = kWheelMinus1;
			else if (dtid.wheel() == 0) wheel = kWheelZero;
			else if (dtid.wheel() == 1) wheel = kWheelPlus1;
			else if (dtid.wheel() == 2) wheel = kWheelPlus2;

			int sector = dtid.sector() - 1;
		
			double resid_x = dt13->global_hitresid(m_layer);
			double resid_dxdz = dt13->global_resslope();

			if (fabs(resid_x) < 10.) {
			   th2f_wheelsector[wheel][sector][kDeltaX]->Fill(qoverpt, resid_x*10.);
			   tprofile_wheelsector[wheel][sector][kDeltaX]->Fill(qoverpt, resid_x*10.);
			   th2f_wheelsector[wheel][sector][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);
			   tprofile_wheelsector[wheel][sector][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);
			}

			// derivatives are in local coordinates, so these should be, too
			resid_x = dt13->hitresid(m_layer);
			resid_dxdz = dt13->resslope();

			// calculate derivative
			TrajectoryStateOnSurface last_tracker_tsos;
			double last_tracker_R = 0.;
			std::vector<TrajectoryMeasurement> measurements = traj->measurements();
			for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
			   TrajectoryStateOnSurface tsos = im->forwardPredictedState();
			   if (tsos.isValid()) {
			      GlobalPoint pos = tsos.globalPosition();
			      if (pos.perp() < 200.  &&  fabs(pos.z()) < 400.) {  // if in tracker (cheap, I know...)
				 if (pos.perp() > last_tracker_R) {
				    last_tracker_tsos = tsos;
				    last_tracker_R = pos.perp();
				 }
			      }
			   }
			}
			if (last_tracker_R > 0.) {
			   FreeTrajectoryState ts_rebuilt(last_tracker_tsos.globalPosition(),
							  last_tracker_tsos.globalMomentum(),
							  last_tracker_tsos.charge(),
							  &*magneticField);

			   double factor = (last_tracker_tsos.globalMomentum().mag() + 1.) / last_tracker_tsos.globalMomentum().mag();
			   FreeTrajectoryState ts_plus1GeV(last_tracker_tsos.globalPosition(),
							   GlobalVector(factor*last_tracker_tsos.globalMomentum().x(), factor*last_tracker_tsos.globalMomentum().y(), factor*last_tracker_tsos.globalMomentum().z()),
							   last_tracker_tsos.charge(),
							   &*magneticField);

			   TrajectoryStateOnSurface extrapolation_rebuilt = propagator->propagate(ts_rebuilt, globalGeometry->idToDet(dt13->localid(m_layer))->surface());
			   TrajectoryStateOnSurface extrapolation_plus1GeV = propagator->propagate(ts_plus1GeV, globalGeometry->idToDet(dt13->localid(m_layer))->surface());

			   if (extrapolation_rebuilt.isValid() && extrapolation_plus1GeV.isValid()) {
			   	double rebuiltx = extrapolation_rebuilt.localPosition().x();
			   	double plus1x = extrapolation_plus1GeV.localPosition().x();

			   	if (fabs(resid_x) < 10.) {
			      		th2f_wheelsector[wheel][sector][kPtErr]->Fill(1./qoverpt, resid_x/(rebuiltx-plus1x)/sqrt(1. + pz*pz/(px*px + py*py))*fabs(qoverpt)*100.);
			      		tprofile_wheelsector[wheel][sector][kPtErr]->Fill(1./qoverpt, resid_x/(rebuiltx-plus1x)/sqrt(1. + pz*pz/(px*px + py*py))*fabs(qoverpt)*100.);

			      		th2f_wheelsector[wheel][sector][kCurvErr]->Fill(qoverpt, resid_x/(rebuiltx-plus1x)*qoverpt/sqrt(px*px + py*py + pz*pz));
			      		tprofile_wheelsector[wheel][sector][kCurvErr]->Fill(qoverpt, resid_x/(rebuiltx-plus1x)*qoverpt/sqrt(px*px + py*py + pz*pz));
			   	}
			   }
			}

		     } // if it's a good segment
		  } // if on our chamber
	       } // if DT

	       if (chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::CSC) {
		  CSCDetId cscid(chamberId->rawId());
		  if (cscid.station() == 1) {
		
		     MuonChamberResidual *csc = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kCSC);

		     if (csc != NULL  &&  csc->numHits() >= m_minCSCHits) {
			int wheel = -1;
			int endcap = -1;
			if (cscid.endcap() == 1  &&  cscid.ring() == 1) { wheel = kWheelMEp11;  endcap = kEndcapMEp11; }
			else if (cscid.endcap() == 1  &&  cscid.ring() == 2) { wheel = kWheelMEp12;  endcap = kEndcapMEp12; }
			else if (cscid.endcap() == 1  &&  cscid.ring() == 3) { wheel = kWheelMEp13;  endcap = kEndcapMEp13; }
			else if (cscid.endcap() == 1  &&  cscid.ring() == 4) { wheel = kWheelMEp14;  endcap = kEndcapMEp14; }
			else if (cscid.endcap() != 1  &&  cscid.ring() == 1) { wheel = kWheelMEm11;  endcap = kEndcapMEm11; }
			else if (cscid.endcap() != 1  &&  cscid.ring() == 2) { wheel = kWheelMEm12;  endcap = kEndcapMEm12; }
			else if (cscid.endcap() != 1  &&  cscid.ring() == 3) { wheel = kWheelMEm13;  endcap = kEndcapMEm13; }
			else if (cscid.endcap() != 1  &&  cscid.ring() == 4) { wheel = kWheelMEm14;  endcap = kEndcapMEm14; }

			int chamber = cscid.chamber() - 1;

			int sector = cscid.chamber() / 3;
			if (cscid.chamber() == 36) sector = 0;

			double resid_x = csc->global_hitresid(m_layer);
			double resid_dxdz = csc->global_resslope();

			if (fabs(resid_x) < 10.) {
			   th2f_wheelsector[wheel][sector][kDeltaX]->Fill(qoverpt, resid_x*10.);
			   tprofile_wheelsector[wheel][sector][kDeltaX]->Fill(qoverpt, resid_x*10.);
			   th2f_wheelsector[wheel][sector][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);
			   tprofile_wheelsector[wheel][sector][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);

			   th2f_endcap[endcap][chamber][kDeltaX]->Fill(qoverpt, resid_x*10.);
			   tprofile_endcap[endcap][chamber][kDeltaX]->Fill(qoverpt, resid_x*10.);
			   th2f_endcap[endcap][chamber][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);
			   tprofile_endcap[endcap][chamber][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);

			   if (cscid.chamber() % 2 == 0) {
			      th2f_evens[endcap][kDeltaX]->Fill(qoverpt, resid_x*10.);
			      tprofile_evens[endcap][kDeltaX]->Fill(qoverpt, resid_x*10.);
			      th2f_evens[endcap][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);
			      tprofile_evens[endcap][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);
			   }
			   else {
			      th2f_odds[endcap][kDeltaX]->Fill(qoverpt, resid_x*10.);
			      tprofile_odds[endcap][kDeltaX]->Fill(qoverpt, resid_x*10.);
			      th2f_odds[endcap][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);
			      tprofile_odds[endcap][kDeltaDxDz]->Fill(qoverpt, resid_dxdz*1000.);
			   }
			}

			// derivatives are in local coordinates, so these should be, too
			resid_x = csc->hitresid(m_layer);
			resid_dxdz = csc->resslope();

			// calculate derivative
			TrajectoryStateOnSurface last_tracker_tsos;
			double last_tracker_absZ = 0.;
			std::vector<TrajectoryMeasurement> measurements = traj->measurements();
			for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
			   TrajectoryStateOnSurface tsos = im->forwardPredictedState();
			   if (tsos.isValid()) {
			      GlobalPoint pos = tsos.globalPosition();
			      if (pos.perp() < 200.  &&  fabs(pos.z()) < 400.) {
				 if (fabs(pos.z()) > last_tracker_absZ) {
				    last_tracker_tsos = tsos;
				    last_tracker_absZ = fabs(pos.z());
				 }
			      }
			   }
			}
			if (last_tracker_absZ > 0.) {
			   FreeTrajectoryState ts_rebuilt(last_tracker_tsos.globalPosition(),
							  last_tracker_tsos.globalMomentum(),
							  last_tracker_tsos.charge(),
							  &*magneticField);

			   double factor = (last_tracker_tsos.globalMomentum().mag() + 1.) / last_tracker_tsos.globalMomentum().mag();
			   FreeTrajectoryState ts_plus1GeV(last_tracker_tsos.globalPosition(),
							   GlobalVector(factor*last_tracker_tsos.globalMomentum().x(), factor*last_tracker_tsos.globalMomentum().y(), factor*last_tracker_tsos.globalMomentum().z()),
							   last_tracker_tsos.charge(),
							   &*magneticField);

			   TrajectoryStateOnSurface extrapolation_rebuilt = propagator->propagate(ts_rebuilt, globalGeometry->idToDet(csc->localid(m_layer))->surface());
			   TrajectoryStateOnSurface extrapolation_plus1GeV = propagator->propagate(ts_plus1GeV, globalGeometry->idToDet(csc->localid(m_layer))->surface());

			   if (extrapolation_rebuilt.isValid() && extrapolation_plus1GeV.isValid()) {

			   	double rebuiltx = extrapolation_rebuilt.localPosition().x();
			   	double plus1x = extrapolation_plus1GeV.localPosition().x();

			   	if (fabs(resid_x) < 10.) {
			      		double pterroverpt = resid_x/(rebuiltx-plus1x)/sqrt(1. + pz*pz/(px*px + py*py))*fabs(qoverpt)*100.;
			      		double curverror = resid_x/(rebuiltx-plus1x)*qoverpt/sqrt(px*px + py*py + pz*pz);

			      		th2f_wheelsector[wheel][sector][kPtErr]->Fill(1./qoverpt, pterroverpt);
			      		tprofile_wheelsector[wheel][sector][kPtErr]->Fill(1./qoverpt, pterroverpt);
			      		th2f_wheelsector[wheel][sector][kCurvErr]->Fill(qoverpt, curverror);
			      		tprofile_wheelsector[wheel][sector][kCurvErr]->Fill(qoverpt, curverror);

			      		th2f_endcap[endcap][chamber][kPtErr]->Fill(1./qoverpt, pterroverpt);
			      		tprofile_endcap[endcap][chamber][kPtErr]->Fill(1./qoverpt, pterroverpt);
			      		th2f_endcap[endcap][chamber][kCurvErr]->Fill(qoverpt, curverror);
			      		tprofile_endcap[endcap][chamber][kCurvErr]->Fill(qoverpt, curverror);

			      		if (cscid.chamber() % 2 == 0) {
				 		th2f_evens[endcap][kPtErr]->Fill(1./qoverpt, pterroverpt);
				 		tprofile_evens[endcap][kPtErr]->Fill(1./qoverpt, pterroverpt);
				 		th2f_evens[endcap][kCurvErr]->Fill(qoverpt, curverror);
				 		tprofile_evens[endcap][kCurvErr]->Fill(qoverpt, curverror);
			      		}
			      		else {
				 		th2f_odds[endcap][kPtErr]->Fill(1./qoverpt, pterroverpt);
				 		tprofile_odds[endcap][kPtErr]->Fill(1./qoverpt, pterroverpt);
				 		th2f_odds[endcap][kCurvErr]->Fill(qoverpt, curverror);
				 		tprofile_odds[endcap][kCurvErr]->Fill(qoverpt, curverror);
			      		}
			   	}
			   }
			}
		     } // if it's a good segment
		  } // if on our chamber
	       } // if CSC

	    } // end loop over chamberIds
	 } // end if refit is okay
      } // end if track pT is within range
   } // end loop over tracks
}

void AlignmentMonitorMuonVsCurvature::afterAlignment(const edm::EventSetup &iSetup) {}

//
// constructors and destructor
//

// AlignmentMonitorMuonVsCurvature::AlignmentMonitorMuonVsCurvature(const AlignmentMonitorMuonVsCurvature& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorMuonVsCurvature& AlignmentMonitorMuonVsCurvature::operator=(const AlignmentMonitorMuonVsCurvature& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorMuonVsCurvature temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// const member functions
//

//
// static member functions
//

//
// SEAL definitions
//

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorMuonVsCurvature, "AlignmentMonitorMuonVsCurvature");
