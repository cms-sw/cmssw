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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

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

class MuonIdDQM : public DQMEDAnalyzer {
   public:
      explicit MuonIdDQM(const edm::ParameterSet&);
      ~MuonIdDQM();

      /* Operations */
      void analyze(const edm::Event&, const edm::EventSetup&);
      void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

   private:
      virtual void Fill(MonitorElement*, float);

      // ----------member data ---------------------------
      edm::EDGetTokenT<reco::MuonCollection> inputMuonCollection_;
      edm::EDGetTokenT<DTRecSegment4DCollection> inputDTRecSegment4DCollection_;
      edm::EDGetTokenT<CSCSegmentCollection> inputCSCSegmentCollection_;
      bool useTrackerMuons_;
      bool useGlobalMuons_;
      bool useTrackerMuonsNotGlobalMuons_;
      bool useGlobalMuonsNotTrackerMuons_;
      std::string baseFolder_;

      edm::Handle<reco::MuonCollection> muonCollectionH_;
      edm::Handle<DTRecSegment4DCollection> dtSegmentCollectionH_;
      edm::Handle<CSCSegmentCollection> cscSegmentCollectionH_;
      edm::ESHandle<GlobalTrackingGeometry> geometry_;

      // trackerMuon == 0; globalMuon == 1
      MonitorElement* hNumChambers[4];
      MonitorElement* hNumMatches[4];
      MonitorElement* hNumChambersNoRPC[4];

      // by station
      MonitorElement* hDTNumSegments[4][4];
      MonitorElement* hDTDx[4][4];
      MonitorElement* hDTPullx[4][4];
      MonitorElement* hDTDdXdZ[4][4];
      MonitorElement* hDTPulldXdZ[4][4];
      MonitorElement* hDTDy[4][3];
      MonitorElement* hDTPully[4][3];
      MonitorElement* hDTDdYdZ[4][3];
      MonitorElement* hDTPulldYdZ[4][3];
      MonitorElement* hCSCNumSegments[4][4];
      MonitorElement* hCSCDx[4][4];
      MonitorElement* hCSCPullx[4][4];
      MonitorElement* hCSCDdXdZ[4][4];
      MonitorElement* hCSCPulldXdZ[4][4];
      MonitorElement* hCSCDy[4][4];
      MonitorElement* hCSCPully[4][4];
      MonitorElement* hCSCDdYdZ[4][4];
      MonitorElement* hCSCPulldYdZ[4][4];

      // segment matching "efficiency"
      MonitorElement* hSegmentIsAssociatedBool;
};

#endif
