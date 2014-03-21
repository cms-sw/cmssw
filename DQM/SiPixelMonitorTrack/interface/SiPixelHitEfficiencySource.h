#ifndef SiPixelHitEfficiencySource_H
#define SiPixelHitEfficiencySource_H

// Package: SiPixelMonitorTrack
// Class:   SiPixelHitEfficiencySource
// 
// class SiPixelHitEfficiencySource SiPixelHitEfficiencySource.h 
//       DQM/SiPixelMonitorTrack/interface/SiPixelHitEfficiencySource.h
// 
// Description:    <one line class summary>
// Implementation: <Notes on implementation>
//
//
// Original Authors: Romain Rougny & Luca Mucibello
//         Created: Mar Nov 10 13:29:00 CET 2009


#include <boost/cstdint.hpp>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelHitEfficiencyModule.h"

//Files added for monitoring track quantities
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class SiPixelHitEfficiencySource : public edm::EDAnalyzer {
  public:
    explicit SiPixelHitEfficiencySource(const edm::ParameterSet&);
            ~SiPixelHitEfficiencySource();

    virtual void beginJob();
    virtual void endJob(void);
    virtual void beginRun(const edm::Run& r, edm::EventSetup const& iSetup);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private: 
    edm::ParameterSet pSet_; 
    edm::InputTag src_; 
    // edm::InputTag tracksrc_;
    edm::EDGetTokenT<reco::VertexCollection> vertexCollectionToken_;
    edm::EDGetTokenT<TrajTrackAssociationCollection> tracksrc_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > clusterCollectionToken_;
    
    edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerEventToken_;

    bool applyEdgeCut_;
    double nSigma_EdgeCut_;
    
    DQMStore* dbe_; 

    bool debug_; 
    bool modOn; 
    //barrel:
    bool ladOn, layOn, phiOn;
    //forward:
    bool ringOn, bladeOn, diskOn; 

    bool firstRun;
    
    std::map<uint32_t, SiPixelHitEfficiencyModule*> theSiPixelStructure;
    
    int nmissing,nvalid; 
    
    int nvtx_;
    int vtxntrk_;
    double vtxD0_;
    double vtxX_;
    double vtxY_;
    double vtxZ_;
    double vtxndof_;
    double vtxchi2_;

    bool isUpgrade;

};

#endif
