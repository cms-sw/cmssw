// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorSegmentDifferences
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Mon Nov 12 13:30:14 CST 2007
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

#include <sstream>

// user include files

// 
// class definition
// 

class AlignmentMonitorSegmentDifferences: public AlignmentMonitorBase {
public:
  AlignmentMonitorSegmentDifferences(const edm::ParameterSet& cfg);
  ~AlignmentMonitorSegmentDifferences() {};

  void book();
  void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
  void afterAlignment(const edm::EventSetup &iSetup);

private:
  double m_minTrackPt;
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  int m_minDT13Hits;
  int m_minDT2Hits;
  int m_minCSCHits;

  // wheel, sector, stationdiff
  TProfile *m_dt13_resid[5][12][3];
  TProfile *m_dt13_slope[5][12][3];
  TProfile *m_dt2_resid[5][12][2];
  TProfile *m_dt2_slope[5][12][2];
  TH1F *m_posdt13_resid[5][12][3];
  TH1F *m_posdt13_slope[5][12][3];
  TH1F *m_posdt2_resid[5][12][2];
  TH1F *m_posdt2_slope[5][12][2];
  TH1F *m_negdt13_resid[5][12][3];
  TH1F *m_negdt13_slope[5][12][3];
  TH1F *m_negdt2_resid[5][12][2];
  TH1F *m_negdt2_slope[5][12][2];

  // endcap, chamber, stationdiff
  TProfile *m_cscouter_resid[2][36][2];
  TProfile *m_cscouter_slope[2][36][2];
  TProfile *m_cscinner_resid[2][18][3];
  TProfile *m_cscinner_slope[2][18][3];
  TH1F *m_poscscouter_resid[2][36][2];
  TH1F *m_poscscouter_slope[2][36][2];
  TH1F *m_poscscinner_resid[2][18][3];
  TH1F *m_poscscinner_slope[2][18][3];
  TH1F *m_negcscouter_resid[2][36][2];
  TH1F *m_negcscouter_slope[2][36][2];
  TH1F *m_negcscinner_resid[2][18][3];
  TH1F *m_negcscinner_slope[2][18][3];
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

AlignmentMonitorSegmentDifferences::AlignmentMonitorSegmentDifferences(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorSegmentDifferences")
   , m_minTrackPt(cfg.getParameter<double>("minTrackPt"))
   , m_minTrackerHits(cfg.getParameter<int>("minTrackerHits"))
   , m_maxTrackerRedChi2(cfg.getParameter<double>("maxTrackerRedChi2"))
   , m_allowTIDTEC(cfg.getParameter<bool>("allowTIDTEC"))
   , m_minDT13Hits(cfg.getParameter<int>("minDT13Hits"))
   , m_minDT2Hits(cfg.getParameter<int>("minDT2Hits"))
   , m_minCSCHits(cfg.getParameter<int>("minCSCHits"))
{
}

void AlignmentMonitorSegmentDifferences::book() {
   for (int wheel = -2;  wheel <= +2;  wheel++) {
      for (int sector = 1;  sector <= 12;  sector++) {
	 char num[3];
	 num[0] = ('0' + (sector / 10));
	 num[1] = ('0' + (sector % 10));
	 num[2] = 0;
	 
	 std::string wheelletter;
	 if (wheel == -2) wheelletter = "A";
	 else if (wheel == -1) wheelletter = "B";
	 else if (wheel ==  0) wheelletter = "C";
	 else if (wheel == +1) wheelletter = "D";
	 else if (wheel == +2) wheelletter = "E";

	 std::string name, pos, neg;

	 name = (std::string("dt13_resid_") + wheelletter + std::string("_") + std::string(num) + std::string("_12"));
	 m_dt13_resid[wheel+2][sector-1][0] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_posdt13_resid[wheel+2][sector-1][0] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negdt13_resid[wheel+2][sector-1][0] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("dt13_resid_") + wheelletter + std::string("_") + std::string(num) + std::string("_23"));
	 m_dt13_resid[wheel+2][sector-1][1] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_posdt13_resid[wheel+2][sector-1][1] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negdt13_resid[wheel+2][sector-1][1] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("dt13_resid_") + wheelletter + std::string("_") + std::string(num) + std::string("_34"));
	 m_dt13_resid[wheel+2][sector-1][2] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_posdt13_resid[wheel+2][sector-1][2] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negdt13_resid[wheel+2][sector-1][2] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("dt2_resid_") + wheelletter + std::string("_") + std::string(num) + std::string("_12"));
	 m_dt2_resid[wheel+2][sector-1][0] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -200., 200., " ");
	 pos = std::string("pos") + name;
	 m_posdt2_resid[wheel+2][sector-1][0] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -20., 20.);
	 neg = std::string("neg") + name;
	 m_negdt2_resid[wheel+2][sector-1][0] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -20., 20.);

	 name = (std::string("dt2_resid_") + wheelletter + std::string("_") + std::string(num) + std::string("_23"));
	 m_dt2_resid[wheel+2][sector-1][1] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -200., 200., " ");
	 pos = std::string("pos") + name;
	 m_posdt2_resid[wheel+2][sector-1][1] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -20., 20.);
	 neg = std::string("neg") + name;
	 m_negdt2_resid[wheel+2][sector-1][1] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -20., 20.);

	 name = (std::string("dt13_slope_") + wheelletter + std::string("_") + std::string(num) + std::string("_12"));
	 m_dt13_slope[wheel+2][sector-1][0] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_posdt13_slope[wheel+2][sector-1][0] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negdt13_slope[wheel+2][sector-1][0] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("dt13_slope_") + wheelletter + std::string("_") + std::string(num) + std::string("_23"));
	 m_dt13_slope[wheel+2][sector-1][1] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_posdt13_slope[wheel+2][sector-1][1] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negdt13_slope[wheel+2][sector-1][1] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("dt13_slope_") + wheelletter + std::string("_") + std::string(num) + std::string("_34"));
	 m_dt13_slope[wheel+2][sector-1][2] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_posdt13_slope[wheel+2][sector-1][2] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negdt13_slope[wheel+2][sector-1][2] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("dt2_slope_") + wheelletter + std::string("_") + std::string(num) + std::string("_12"));
	 m_dt2_slope[wheel+2][sector-1][0] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -1000., 1000., " ");
	 pos = std::string("pos") + name;
	 m_posdt2_slope[wheel+2][sector-1][0] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -100., 100.);
	 neg = std::string("neg") + name;
	 m_negdt2_slope[wheel+2][sector-1][0] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -100., 100.);

	 name = (std::string("dt2_slope_") + wheelletter + std::string("_") + std::string(num) + std::string("_23"));
	 m_dt2_slope[wheel+2][sector-1][1] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -1000., 1000., " ");
	 pos = std::string("pos") + name;
	 m_posdt2_slope[wheel+2][sector-1][1] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -100., 100.);
	 neg = std::string("neg") + name;
	 m_negdt2_slope[wheel+2][sector-1][1] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -100., 100.);
      }
   }

   for (int endcap = 1;  endcap <= 2;  endcap++) {
      for (int chamber = 1;  chamber <= 36;  chamber++) {
	 char num[3];
	 num[0] = ('0' + (chamber / 10));
	 num[1] = ('0' + (chamber % 10));
	 num[2] = 0;

	 std::string endcapletter;
	 if (endcap == 1) endcapletter = "p";
	 else if (endcap == 2) endcapletter = "m";

	 std::string name, pos, neg;

	 name = (std::string("cscouter_resid_") + endcapletter + std::string("_") + std::string(num) + std::string("_12"));
	 m_cscouter_resid[endcap-1][chamber-1][0] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscouter_resid[endcap-1][chamber-1][0] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscouter_resid[endcap-1][chamber-1][0] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("cscouter_resid_") + endcapletter + std::string("_") + std::string(num) + std::string("_23"));
	 m_cscouter_resid[endcap-1][chamber-1][1] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscouter_resid[endcap-1][chamber-1][1] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscouter_resid[endcap-1][chamber-1][1] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("cscouter_slope_") + endcapletter + std::string("_") + std::string(num) + std::string("_12"));
	 m_cscouter_slope[endcap-1][chamber-1][0] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscouter_slope[endcap-1][chamber-1][0] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscouter_slope[endcap-1][chamber-1][0] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("cscouter_slope_") + endcapletter + std::string("_") + std::string(num) + std::string("_23"));
	 m_cscouter_slope[endcap-1][chamber-1][1] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscouter_slope[endcap-1][chamber-1][1] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscouter_slope[endcap-1][chamber-1][1] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);
      }

      for (int chamber = 1;  chamber <= 18;  chamber++) {
	 char num[3];
	 num[0] = ('0' + (chamber / 10));
	 num[1] = ('0' + (chamber % 10));
	 num[2] = 0;

	 std::string endcapletter;
	 if (endcap == 1) endcapletter = "p";
	 else if (endcap == 2) endcapletter = "m";

	 std::string name, pos, neg;

	 name = (std::string("cscinner_resid_") + endcapletter + std::string("_") + std::string(num) + std::string("_12"));
	 m_cscinner_resid[endcap-1][chamber-1][0] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscinner_resid[endcap-1][chamber-1][0] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscinner_resid[endcap-1][chamber-1][0] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("cscinner_resid_") + endcapletter + std::string("_") + std::string(num) + std::string("_23"));
	 m_cscinner_resid[endcap-1][chamber-1][1] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscinner_resid[endcap-1][chamber-1][1] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscinner_resid[endcap-1][chamber-1][1] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("cscinner_resid_") + endcapletter + std::string("_") + std::string(num) + std::string("_34"));
	 m_cscinner_resid[endcap-1][chamber-1][2] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscinner_resid[endcap-1][chamber-1][2] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscinner_resid[endcap-1][chamber-1][2] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("cscinner_slope_") + endcapletter + std::string("_") + std::string(num) + std::string("_12"));
	 m_cscinner_slope[endcap-1][chamber-1][0] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscinner_slope[endcap-1][chamber-1][0] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscinner_slope[endcap-1][chamber-1][0] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("cscinner_slope_") + endcapletter + std::string("_") + std::string(num) + std::string("_23"));
	 m_cscinner_slope[endcap-1][chamber-1][1] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscinner_slope[endcap-1][chamber-1][1] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscinner_slope[endcap-1][chamber-1][1] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);

	 name = (std::string("cscinner_slope_") + endcapletter + std::string("_") + std::string(num) + std::string("_34"));
	 m_cscinner_slope[endcap-1][chamber-1][2] = bookProfile("/iterN/", name.c_str(), name.c_str(), 20, -1./m_minTrackPt, 1./m_minTrackPt, 1, -100., 100., " ");
	 pos = std::string("pos") + name;
	 m_poscscinner_slope[endcap-1][chamber-1][2] = book1D("/iterN/", pos.c_str(), pos.c_str(), 100, -10., 10.);
	 neg = std::string("neg") + name;
	 m_negcscinner_slope[endcap-1][chamber-1][2] = book1D("/iterN/", neg.c_str(), neg.c_str(), 100, -10., 10.);
      }
   }
}

void AlignmentMonitorSegmentDifferences::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& trajtracks) {
  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack) {
    const Trajectory* traj = (*trajtrack).first;
    const reco::Track* track = (*trajtrack).second;

    if (track->pt() > m_minTrackPt) {
      double qoverpt = (track->charge() > 0 ? 1. : -1.) / track->pt();
      MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, pNavigator(), 1000.);

      if (muonResidualsFromTrack.trackerNumHits() >= m_minTrackerHits  &&  muonResidualsFromTrack.trackerRedChi2() < m_maxTrackerRedChi2  &&  (m_allowTIDTEC  ||  !muonResidualsFromTrack.contains_TIDTEC())) {
	std::vector<DetId> chamberIds = muonResidualsFromTrack.chamberIds();

	for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId) {
	  if (chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::DT) {
	    MuonChamberResidual *dt13 = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kDT13);
	    MuonChamberResidual *dt2 = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kDT2);

	    if (dt13 != NULL  &&  dt13->numHits() >= m_minDT13Hits) {
	       DTChamberId thisid(chamberId->rawId());

	       for (std::vector<DetId>::const_iterator otherId = chamberIds.begin();  otherId != chamberIds.end();  ++otherId) {
		  if (otherId->det() == DetId::Muon  &&  otherId->subdetId() == MuonSubdetId::DT) {
		     DTChamberId thatid(otherId->rawId());
		     if (thisid.rawId() != thatid.rawId()  &&  thisid.wheel() == thatid.wheel()  &&  thisid.sector() == thatid.sector()) {
			MuonChamberResidual *dt13other = muonResidualsFromTrack.chamberResidual(*otherId, MuonChamberResidual::kDT13);
			if (dt13other != NULL  &&  dt13other->numHits() >= m_minDT13Hits) {
			   double slopediff = 1000. * (dt13->global_resslope() - dt13other->global_resslope());

			   double length = dt13->chamberAlignable()->surface().toGlobal(LocalPoint(0,0,0)).perp() - dt13other->chamberAlignable()->surface().toGlobal(LocalPoint(0,0,0)).perp();
			   double residdiff = 10. * (dt13->global_residual() + length*dt13->global_resslope() - dt13other->global_residual());

			   if (thisid.station() == 1  &&  thatid.station() == 2) {
			      m_dt13_resid[thisid.wheel()+2][thisid.sector()-1][0]->Fill(qoverpt, residdiff);
			      m_dt13_slope[thisid.wheel()+2][thisid.sector()-1][0]->Fill(qoverpt, slopediff);
			      if (qoverpt > 0) {
				 m_posdt13_resid[thisid.wheel()+2][thisid.sector()-1][0]->Fill(residdiff);
				 m_posdt13_slope[thisid.wheel()+2][thisid.sector()-1][0]->Fill(slopediff);
			      }
			      else {
				 m_negdt13_resid[thisid.wheel()+2][thisid.sector()-1][0]->Fill(residdiff);
				 m_negdt13_slope[thisid.wheel()+2][thisid.sector()-1][0]->Fill(slopediff);
			      }
			   }
			   else if (thisid.station() == 2  &&  thatid.station() == 3) {
			      m_dt13_resid[thisid.wheel()+2][thisid.sector()-1][1]->Fill(qoverpt, residdiff);
			      m_dt13_slope[thisid.wheel()+2][thisid.sector()-1][1]->Fill(qoverpt, slopediff);
			      if (qoverpt > 0) {
				 m_posdt13_resid[thisid.wheel()+2][thisid.sector()-1][1]->Fill(residdiff);
				 m_posdt13_slope[thisid.wheel()+2][thisid.sector()-1][1]->Fill(slopediff);
			      }
			      else {
				 m_negdt13_resid[thisid.wheel()+2][thisid.sector()-1][1]->Fill(residdiff);
				 m_negdt13_slope[thisid.wheel()+2][thisid.sector()-1][1]->Fill(slopediff);
			      }
			   }
			   else if (thisid.station() == 3  &&  thatid.station() == 4) {
			      m_dt13_resid[thisid.wheel()+2][thisid.sector()-1][2]->Fill(qoverpt, residdiff);
			      m_dt13_slope[thisid.wheel()+2][thisid.sector()-1][2]->Fill(qoverpt, slopediff);
			      if (qoverpt > 0) {
				 m_posdt13_resid[thisid.wheel()+2][thisid.sector()-1][2]->Fill(residdiff);
				 m_posdt13_slope[thisid.wheel()+2][thisid.sector()-1][2]->Fill(slopediff);
			      }
			      else {
				 m_negdt13_resid[thisid.wheel()+2][thisid.sector()-1][2]->Fill(residdiff);
				 m_negdt13_slope[thisid.wheel()+2][thisid.sector()-1][2]->Fill(slopediff);
			      }
			   }

			} // end other numhits
		     } // end this near other
		  } // end other is DT
	       } // end loop over other

	    } // end if DT13

	    if (dt2 != NULL  &&  dt2->numHits() >= m_minDT2Hits) {
	       DTChamberId thisid(chamberId->rawId());

	       for (std::vector<DetId>::const_iterator otherId = chamberIds.begin();  otherId != chamberIds.end();  ++otherId) {
		  if (otherId->det() == DetId::Muon  &&  otherId->subdetId() == MuonSubdetId::DT) {
		     DTChamberId thatid(otherId->rawId());
		     if (thisid.rawId() != thatid.rawId()  &&  thisid.wheel() == thatid.wheel()  &&  thisid.sector() == thatid.sector()) {
			MuonChamberResidual *dt2other = muonResidualsFromTrack.chamberResidual(*otherId, MuonChamberResidual::kDT2);
			if (dt2other != NULL  &&  dt2other->numHits() >= m_minDT2Hits) {
			   double slopediff = 1000. * (dt2->global_resslope() - dt2other->global_resslope());

			   double length = dt2->chamberAlignable()->surface().toGlobal(LocalPoint(0,0,0)).perp() - dt2other->chamberAlignable()->surface().toGlobal(LocalPoint(0,0,0)).perp();
			   double residdiff = 10. * (dt2->global_residual() + length*dt2->global_resslope() - dt2other->global_residual());

			   if (thisid.station() == 1  &&  thatid.station() == 2) {
			      m_dt2_resid[thisid.wheel()+2][thisid.sector()-1][0]->Fill(qoverpt, residdiff);
			      m_dt2_slope[thisid.wheel()+2][thisid.sector()-1][0]->Fill(qoverpt, slopediff);
			      if (qoverpt > 0) {
				 m_posdt2_resid[thisid.wheel()+2][thisid.sector()-1][0]->Fill(residdiff);
				 m_posdt2_slope[thisid.wheel()+2][thisid.sector()-1][0]->Fill(slopediff);
			      }
			      else {
				 m_negdt2_resid[thisid.wheel()+2][thisid.sector()-1][0]->Fill(residdiff);
				 m_negdt2_slope[thisid.wheel()+2][thisid.sector()-1][0]->Fill(slopediff);
			      }
			   }
			   else if (thisid.station() == 2  &&  thatid.station() == 3) {
			      m_dt2_resid[thisid.wheel()+2][thisid.sector()-1][1]->Fill(qoverpt, residdiff);
			      m_dt2_slope[thisid.wheel()+2][thisid.sector()-1][1]->Fill(qoverpt, slopediff);
			      if (qoverpt > 0) {
				 m_posdt2_resid[thisid.wheel()+2][thisid.sector()-1][1]->Fill(residdiff);
				 m_posdt2_slope[thisid.wheel()+2][thisid.sector()-1][1]->Fill(slopediff);
			      }
			      else {
				 m_negdt2_resid[thisid.wheel()+2][thisid.sector()-1][1]->Fill(residdiff);
				 m_negdt2_slope[thisid.wheel()+2][thisid.sector()-1][1]->Fill(slopediff);
			      }
			   }
			   else if (thisid.station() == 3  &&  thatid.station() == 4) {
			      m_dt2_resid[thisid.wheel()+2][thisid.sector()-1][2]->Fill(qoverpt, residdiff);
			      m_dt2_slope[thisid.wheel()+2][thisid.sector()-1][2]->Fill(qoverpt, slopediff);
			      if (qoverpt > 0) {
				 m_posdt2_resid[thisid.wheel()+2][thisid.sector()-1][2]->Fill(residdiff);
				 m_posdt2_slope[thisid.wheel()+2][thisid.sector()-1][2]->Fill(slopediff);
			      }
			      else {
				 m_negdt2_resid[thisid.wheel()+2][thisid.sector()-1][2]->Fill(residdiff);
				 m_negdt2_slope[thisid.wheel()+2][thisid.sector()-1][2]->Fill(slopediff);
			      }
			   }

			} // end other numhits
		     } // end this near other
		  } // end other is DT
	       } // end loop over other

	    } // end if DT2
	  } // end if DT

	  else if (chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::CSC) {
	    MuonChamberResidual *csc = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kCSC);

	    if (csc->numHits() >= m_minCSCHits) {
	       CSCDetId thisid(chamberId->rawId());

	       for (std::vector<DetId>::const_iterator otherId = chamberIds.begin();  otherId != chamberIds.end();  ++otherId) {
		  if (otherId->det() == DetId::Muon  &&  otherId->subdetId() == MuonSubdetId::CSC) {
		     CSCDetId thatid(otherId->rawId());
		     if (thisid.rawId() != thatid.rawId()  &&  thisid.endcap() == thatid.endcap()) {
			MuonChamberResidual *cscother = muonResidualsFromTrack.chamberResidual(*otherId, MuonChamberResidual::kCSC);
			if (cscother != NULL  &&  cscother->numHits() >= m_minCSCHits) {
			   double slopediff = 1000. * (csc->global_resslope() - cscother->global_resslope());

			   double length = csc->chamberAlignable()->surface().toGlobal(LocalPoint(0,0,0)).z() - cscother->chamberAlignable()->surface().toGlobal(LocalPoint(0,0,0)).z();
			   double residdiff = 10. * (csc->global_residual() + length*csc->global_resslope() - cscother->global_residual());

			   int thischamber = thisid.chamber();
			   int thisring = thisid.ring();
			   if (thisid.station() == 1  &&  (thisring == 1  ||  thisring == 4)) {
			      thischamber = (thischamber - 1) / 2 + 1;
			      thisring = 1;
			   }

			   if (thisring == thatid.ring()  &&  thischamber == thatid.chamber()) {
			      if (thisring == 2  &&  thisid.station() == 1  &&  thatid.station() == 2) {
				 m_cscouter_resid[thisid.endcap()-1][thischamber-1][0]->Fill(qoverpt, residdiff);
				 m_cscouter_slope[thisid.endcap()-1][thischamber-1][0]->Fill(qoverpt, slopediff);
				 if (qoverpt > 0) {
				    m_poscscouter_resid[thisid.endcap()-1][thischamber-1][0]->Fill(residdiff);
				    m_poscscouter_slope[thisid.endcap()-1][thischamber-1][0]->Fill(slopediff);
				 }
				 else {
				    m_negcscouter_resid[thisid.endcap()-1][thischamber-1][0]->Fill(residdiff);
				    m_negcscouter_slope[thisid.endcap()-1][thischamber-1][0]->Fill(slopediff);
				 }
			      }
			      else if (thisring == 2  &&  thisid.station() == 2  &&  thatid.station() == 3) {
				 m_cscouter_resid[thisid.endcap()-1][thischamber-1][1]->Fill(qoverpt, residdiff);
				 m_cscouter_slope[thisid.endcap()-1][thischamber-1][1]->Fill(qoverpt, slopediff);
				 if (qoverpt > 0) {
				    m_poscscouter_resid[thisid.endcap()-1][thischamber-1][1]->Fill(residdiff);
				    m_poscscouter_slope[thisid.endcap()-1][thischamber-1][1]->Fill(slopediff);
				 }
				 else {
				    m_negcscouter_resid[thisid.endcap()-1][thischamber-1][1]->Fill(residdiff);
				    m_negcscouter_slope[thisid.endcap()-1][thischamber-1][1]->Fill(slopediff);
				 }
			      }
			      else if (thisring == 1  &&  thisid.station() == 1  &&  thatid.station() == 2) {
				 m_cscinner_resid[thisid.endcap()-1][thischamber-1][0]->Fill(qoverpt, residdiff);
				 m_cscinner_slope[thisid.endcap()-1][thischamber-1][0]->Fill(qoverpt, slopediff);
				 if (qoverpt > 0) {
				    m_poscscinner_resid[thisid.endcap()-1][thischamber-1][0]->Fill(residdiff);
				    m_poscscinner_slope[thisid.endcap()-1][thischamber-1][0]->Fill(slopediff);
				 }
				 else {
				    m_negcscinner_resid[thisid.endcap()-1][thischamber-1][0]->Fill(residdiff);
				    m_negcscinner_slope[thisid.endcap()-1][thischamber-1][0]->Fill(slopediff);
				 }
			      }
			      else if (thisring == 1  &&  thisid.station() == 2  &&  thatid.station() == 3) {
				 m_cscinner_resid[thisid.endcap()-1][thischamber-1][1]->Fill(qoverpt, residdiff);
				 m_cscinner_slope[thisid.endcap()-1][thischamber-1][1]->Fill(qoverpt, slopediff);
				 if (qoverpt > 0) {
				    m_poscscinner_resid[thisid.endcap()-1][thischamber-1][1]->Fill(residdiff);
				    m_poscscinner_slope[thisid.endcap()-1][thischamber-1][1]->Fill(slopediff);
				 }
				 else {
				    m_negcscinner_resid[thisid.endcap()-1][thischamber-1][1]->Fill(residdiff);
				    m_negcscinner_slope[thisid.endcap()-1][thischamber-1][1]->Fill(slopediff);
				 }
			      }
			      else if (thisring == 1  &&  thisid.station() == 3  &&  thatid.station() == 4) {
				 m_cscinner_resid[thisid.endcap()-1][thischamber-1][2]->Fill(qoverpt, residdiff);
				 m_cscinner_slope[thisid.endcap()-1][thischamber-1][2]->Fill(qoverpt, slopediff);
				 if (qoverpt > 0) {
				    m_poscscinner_resid[thisid.endcap()-1][thischamber-1][2]->Fill(residdiff);
				    m_poscscinner_slope[thisid.endcap()-1][thischamber-1][2]->Fill(slopediff);
				 }
				 else {
				    m_negcscinner_resid[thisid.endcap()-1][thischamber-1][2]->Fill(residdiff);
				    m_negcscinner_slope[thisid.endcap()-1][thischamber-1][2]->Fill(slopediff);
				 }
			      }
			   }
			} // end other numhits
		     } // end this near other
		  } // end other is DT
	       } // end loop over other

	    } // end if csc
	  } // end if CSC

	} // end loop over chamberIds
      } // end if refit is okay
    } // end if track pT is within range
  } // end loop over tracks
}

void AlignmentMonitorSegmentDifferences::afterAlignment(const edm::EventSetup &iSetup) {
}

//
// constructors and destructor
//

// AlignmentMonitorSegmentDifferences::AlignmentMonitorSegmentDifferences(const AlignmentMonitorSegmentDifferences& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorSegmentDifferences& AlignmentMonitorSegmentDifferences::operator=(const AlignmentMonitorSegmentDifferences& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorSegmentDifferences temp(rhs);
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

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorSegmentDifferences, "AlignmentMonitorSegmentDifferences");
