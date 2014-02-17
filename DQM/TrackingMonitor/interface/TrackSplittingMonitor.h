#ifndef TrackSplittingMonitor_H
#define TrackSplittingMonitor_H
// -*- C++ -*-
//
// Package:    TrackSplittingMonitor
// Class:      TrackSplittingMonitor
// 
/**\class TrackSplittingMonitor TrackSplittingMonitor.cc DQM/TrackingMonitor/src/TrackSplittingMonitor.cc
 Monitoring source for general quantities related to tracks.
 */
// Original Author:  Nhan Tran
//         Created:  Thu 28 22:45:30 CEST 2008
// $Id: TrackSplittingMonitor.h,v 1.1 2012/10/15 13:24:45 threus Exp $

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class DQMStore;
class TrackAnalyzer;
class TProfile;

class TrackSplittingMonitor : public edm::EDAnalyzer {
public:
	explicit TrackSplittingMonitor(const edm::ParameterSet&);
	~TrackSplittingMonitor();
	virtual void beginJob(void);
	virtual void endJob(void);
	
	virtual void analyze(const edm::Event&, const edm::EventSetup&);
	
private:
	void doProfileX(TH2 * th2, MonitorElement* me);
	void doProfileX(MonitorElement * th2m, MonitorElement* me);
	
	
	// ----------member data ---------------------------
	
	//  unsigned int minTracks_;
	
	std::string histname;  //for naming the histograms according to algorithm used
	
	DQMStore * dqmStore_;
	edm::ParameterSet conf_;
	
	edm::ESHandle<TrackerGeometry> theGeometry;
	edm::ESHandle<MagneticField>   theMagField;
	edm::ESHandle<DTGeometry>             dtGeometry;
	edm::ESHandle<CSCGeometry>            cscGeometry;
	edm::ESHandle<RPCGeometry>            rpcGeometry;
	
	edm::InputTag splitTracks_;
	edm::InputTag splitMuons_;
	
	
	bool plotMuons_;
	int pixelHitsPerLeg_;
	int totalHitsPerLeg_;
	double d0Cut_;
	double dzCut_;
	double ptCut_;
	double norchiCut_;
	
	
	// histograms
	MonitorElement* ddxyAbsoluteResiduals_tracker_;
	MonitorElement* ddzAbsoluteResiduals_tracker_;
	MonitorElement* dphiAbsoluteResiduals_tracker_;
	MonitorElement* dthetaAbsoluteResiduals_tracker_;
	MonitorElement* dptAbsoluteResiduals_tracker_;
	MonitorElement* dcurvAbsoluteResiduals_tracker_;

	MonitorElement* ddxyNormalizedResiduals_tracker_;
	MonitorElement* ddzNormalizedResiduals_tracker_;
	MonitorElement* dphiNormalizedResiduals_tracker_;
	MonitorElement* dthetaNormalizedResiduals_tracker_;
	MonitorElement* dptNormalizedResiduals_tracker_;
	MonitorElement* dcurvNormalizedResiduals_tracker_;
	
	MonitorElement* ddxyAbsoluteResiduals_global_;
	MonitorElement* ddzAbsoluteResiduals_global_;
	MonitorElement* dphiAbsoluteResiduals_global_;
	MonitorElement* dthetaAbsoluteResiduals_global_;
	MonitorElement* dptAbsoluteResiduals_global_;
	MonitorElement* dcurvAbsoluteResiduals_global_;
	
	MonitorElement* ddxyNormalizedResiduals_global_;
	MonitorElement* ddzNormalizedResiduals_global_;
	MonitorElement* dphiNormalizedResiduals_global_;
	MonitorElement* dthetaNormalizedResiduals_global_;
	MonitorElement* dptNormalizedResiduals_global_;
	MonitorElement* dcurvNormalizedResiduals_global_;
	

};
#endif
