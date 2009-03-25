// -*- C++ -*-
//
// Package:    DQMOffline/Muon
// Class:      MuonIdDQM
// 
/*

 Description:  Makes and fills lots of histograms using the various reco::Muon
               methods. All code is adapted from Validation/MuonIdentification


*/
//
// Original Author:  Jacob Ribnik
//         Created:  Wed Apr 18 13:48:08 CDT 2007
// $Id: MuonIdDQM.h,v 1.1 2008/11/04 09:23:23 jribnik Exp $
//
//

#ifndef DQMOffline_Muon_MuonIdDQM_h
#define DQMOffline_Muon_MuonIdDQM_h

// system include files
#include <string>

// user include files
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

class MuonIdDQM : public edm::EDAnalyzer {
   public:
      explicit MuonIdDQM(const edm::ParameterSet&);
      ~MuonIdDQM();

   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      virtual void Fill(MonitorElement*, float);

      DQMStore* dbe_;

      // ----------member data ---------------------------
      edm::InputTag inputMuonCollection_;
      edm::InputTag inputDTRecSegment4DCollection_;
      edm::InputTag inputCSCSegmentCollection_;
      bool useTrackerMuons_;
      bool useGlobalMuons_;
      std::string baseFolder_;

      edm::Handle<reco::MuonCollection> muonCollectionH_;
      edm::Handle<DTRecSegment4DCollection> dtSegmentCollectionH_;
      edm::Handle<CSCSegmentCollection> cscSegmentCollectionH_;
      edm::ESHandle<GlobalTrackingGeometry> geometry_;

      // trackerMuon == 0; globalMuon == 1
      MonitorElement* hNumChambers[2];
      MonitorElement* hNumMatches[2];

      // by station
      MonitorElement* hDTNumSegments[2][4];
      MonitorElement* hDTDx[2][4];
      MonitorElement* hDTPullx[2][4];
      MonitorElement* hDTDdXdZ[2][4];
      MonitorElement* hDTPulldXdZ[2][4];
      MonitorElement* hDTDy[2][3];
      MonitorElement* hDTPully[2][3];
      MonitorElement* hDTDdYdZ[2][3];
      MonitorElement* hDTPulldYdZ[2][3];
      MonitorElement* hCSCNumSegments[2][4];
      MonitorElement* hCSCDx[2][4];
      MonitorElement* hCSCPullx[2][4];
      MonitorElement* hCSCDdXdZ[2][4];
      MonitorElement* hCSCPulldXdZ[2][4];
      MonitorElement* hCSCDy[2][4];
      MonitorElement* hCSCPully[2][4];
      MonitorElement* hCSCDdYdZ[2][4];
      MonitorElement* hCSCPulldYdZ[2][4];

      // segment matching "efficiency"
      MonitorElement* hSegmentIsAssociatedBool;
};

#endif
