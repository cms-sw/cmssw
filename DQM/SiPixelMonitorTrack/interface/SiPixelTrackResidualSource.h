#ifndef SiPixelTrackResidualSource_H
#define SiPixelTrackResidualSource_H

// Package: SiPixelMonitorTrack
// Class:   SiPixelTrackResidualSource
// 
// class SiPixelTrackResidualSource SiPixelTrackResidualSource.h 
//       DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualSource.h
// 
// Description:    <one line class summary>
// Implementation: <Notes on implementation>
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelTrackResidualSource.h,v 1.1 2008/07/25 20:41:29 schuang Exp $
//
// Updated by: Lukas Wehrli
// for pixel offline DQM 

#include <boost/cstdint.hpp>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualModule.h"

//Files added for monitoring track quantities
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class SiPixelTrackResidualSource : public edm::EDAnalyzer {
  public:
    explicit SiPixelTrackResidualSource(const edm::ParameterSet&);
            ~SiPixelTrackResidualSource();

    virtual void beginJob(edm::EventSetup const& iSetup);
    virtual void endJob(void);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private: 
    edm::ParameterSet pSet_; 
    edm::InputTag src_; 
    edm::InputTag clustersrc_; 
    edm::InputTag tracksrc_; 
    DQMStore* dbe_; 

    bool debug_; 
    bool modOn; 
    //barrel:
    bool ladOn, layOn, phiOn;
    //forward:
    bool ringOn, bladeOn, diskOn; 

    std::map<uint32_t, SiPixelTrackResidualModule*> theSiPixelStructure; 

    MonitorElement* meSubdetResidualX[3];
    MonitorElement* meSubdetResidualY[3];

    MonitorElement* meNofTracks_;
    MonitorElement* meNofTracksInPixVol_;
    MonitorElement* meNofClustersOnTrack_;
    MonitorElement* meNofClustersNotOnTrack_;
    MonitorElement* meClChargeOnTrack_all; 
    MonitorElement* meClChargeOnTrack_bpix; 
    MonitorElement* meClChargeOnTrack_fpix; 
    MonitorElement* meClChargeNotOnTrack_all; 
    MonitorElement* meClChargeNotOnTrack_bpix; 
    MonitorElement* meClChargeNotOnTrack_fpix; 
    MonitorElement* meClSizeOnTrack_all; 
    MonitorElement* meClSizeOnTrack_bpix; 
    MonitorElement* meClSizeOnTrack_fpix; 
    MonitorElement* meClSizeNotOnTrack_all; 
    MonitorElement* meClSizeNotOnTrack_bpix; 
    MonitorElement* meClSizeNotOnTrack_fpix; 

    MonitorElement* meClPosLayer1OnTrack; 
    MonitorElement* meClPosLayer2OnTrack; 
    MonitorElement* meClPosLayer3OnTrack; 
    MonitorElement* meClPosLayer1NotOnTrack; 
    MonitorElement* meClPosLayer2NotOnTrack; 
    MonitorElement* meClPosLayer3NotOnTrack; 

    MonitorElement* meClPosDisk1pzOnTrack; 
    MonitorElement* meClPosDisk2pzOnTrack; 
    MonitorElement* meClPosDisk1mzOnTrack; 
    MonitorElement* meClPosDisk2mzOnTrack; 
    MonitorElement* meClPosDisk1pzNotOnTrack; 
    MonitorElement* meClPosDisk2pzNotOnTrack; 
    MonitorElement* meClPosDisk1mzNotOnTrack; 
    MonitorElement* meClPosDisk2mzNotOnTrack; 
};

#endif
