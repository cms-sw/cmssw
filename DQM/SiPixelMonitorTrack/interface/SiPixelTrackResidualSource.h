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
// $Id: SiPixelTrackResidualSource.h,v 1.11 2013/04/17 09:48:46 itopsisg Exp $
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
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class SiPixelTrackResidualSource : public edm::EDAnalyzer {
  public:
    explicit SiPixelTrackResidualSource(const edm::ParameterSet&);
            ~SiPixelTrackResidualSource();

    virtual void beginJob();
    virtual void endJob(void);
    virtual void beginRun(const edm::Run& r, edm::EventSetup const& iSetup);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    void triplets(double x1,double y1,double z1,double x2,double y2,double z2,double x3,double y3,double z3,
                  double ptsig, double & dc,double & dz, double kap); 
  private: 
    edm::ParameterSet pSet_; 
    edm::InputTag src_; 
    edm::InputTag clustersrc_; 
    edm::InputTag tracksrc_; 
    std::string ttrhbuilder_; 
    DQMStore* dbe_; 

    bool debug_; 
    bool modOn; 
    bool reducedSet;
    //barrel:
    bool ladOn, layOn, phiOn;
    //forward:
    bool ringOn, bladeOn, diskOn; 
    bool isUpgrade;
    double ptminres_;
    bool firstRun;
    int NTotal;
    int NLowProb;
    
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
    MonitorElement* meClChargeOnTrack_layer1; 
    MonitorElement* meClChargeOnTrack_layer2; 
    MonitorElement* meClChargeOnTrack_layer3; 
    MonitorElement* meClChargeOnTrack_layer4;
    MonitorElement* meClChargeOnTrack_diskp1; 
    MonitorElement* meClChargeOnTrack_diskp2; 
    MonitorElement* meClChargeOnTrack_diskp3;
    MonitorElement* meClChargeOnTrack_diskm1; 
    MonitorElement* meClChargeOnTrack_diskm2; 
    MonitorElement* meClChargeOnTrack_diskm3;
    MonitorElement* meClChargeNotOnTrack_all; 
    MonitorElement* meClChargeNotOnTrack_bpix; 
    MonitorElement* meClChargeNotOnTrack_fpix; 
    MonitorElement* meClChargeNotOnTrack_layer1; 
    MonitorElement* meClChargeNotOnTrack_layer2; 
    MonitorElement* meClChargeNotOnTrack_layer3; 
    MonitorElement* meClChargeNotOnTrack_layer4; 
    MonitorElement* meClChargeNotOnTrack_diskp1; 
    MonitorElement* meClChargeNotOnTrack_diskp2; 
    MonitorElement* meClChargeNotOnTrack_diskp3;
    MonitorElement* meClChargeNotOnTrack_diskm1; 
    MonitorElement* meClChargeNotOnTrack_diskm2;
    MonitorElement* meClChargeNotOnTrack_diskm3;
    MonitorElement* meClSizeOnTrack_all; 
    MonitorElement* meClSizeOnTrack_bpix; 
    MonitorElement* meClSizeOnTrack_fpix; 
    MonitorElement* meClSizeOnTrack_layer1; 
    MonitorElement* meClSizeOnTrack_layer2; 
    MonitorElement* meClSizeOnTrack_layer3; 
    MonitorElement* meClSizeOnTrack_layer4; 
    MonitorElement* meClSizeOnTrack_diskp1; 
    MonitorElement* meClSizeOnTrack_diskp2; 
    MonitorElement* meClSizeOnTrack_diskp3;
    MonitorElement* meClSizeOnTrack_diskm1; 
    MonitorElement* meClSizeOnTrack_diskm2; 
    MonitorElement* meClSizeOnTrack_diskm3;
    MonitorElement* meClSizeNotOnTrack_all; 
    MonitorElement* meClSizeNotOnTrack_bpix; 
    MonitorElement* meClSizeNotOnTrack_fpix; 
    MonitorElement* meClSizeNotOnTrack_layer1; 
    MonitorElement* meClSizeNotOnTrack_layer2; 
    MonitorElement* meClSizeNotOnTrack_layer3; 
    MonitorElement* meClSizeNotOnTrack_layer4; 
    MonitorElement* meClSizeNotOnTrack_diskp1; 
    MonitorElement* meClSizeNotOnTrack_diskp2; 
    MonitorElement* meClSizeNotOnTrack_diskp3; 
    MonitorElement* meClSizeNotOnTrack_diskm1; 
    MonitorElement* meClSizeNotOnTrack_diskm2;
    MonitorElement* meClSizeNotOnTrack_diskm3;
    MonitorElement* meClSizeXOnTrack_all; 
    MonitorElement* meClSizeXOnTrack_bpix; 
    MonitorElement* meClSizeXOnTrack_fpix; 
    MonitorElement* meClSizeXOnTrack_layer1; 
    MonitorElement* meClSizeXOnTrack_layer2; 
    MonitorElement* meClSizeXOnTrack_layer3; 
    MonitorElement* meClSizeXOnTrack_layer4; 
    MonitorElement* meClSizeXOnTrack_diskp1; 
    MonitorElement* meClSizeXOnTrack_diskp2; 
    MonitorElement* meClSizeXOnTrack_diskp3; 
    MonitorElement* meClSizeXOnTrack_diskm1; 
    MonitorElement* meClSizeXOnTrack_diskm2; 
    MonitorElement* meClSizeXOnTrack_diskm3; 
    MonitorElement* meClSizeXNotOnTrack_all; 
    MonitorElement* meClSizeXNotOnTrack_bpix; 
    MonitorElement* meClSizeXNotOnTrack_fpix; 
    MonitorElement* meClSizeXNotOnTrack_layer1; 
    MonitorElement* meClSizeXNotOnTrack_layer2; 
    MonitorElement* meClSizeXNotOnTrack_layer3; 
    MonitorElement* meClSizeXNotOnTrack_layer4; 
    MonitorElement* meClSizeXNotOnTrack_diskp1; 
    MonitorElement* meClSizeXNotOnTrack_diskp2; 
    MonitorElement* meClSizeXNotOnTrack_diskp3; 
    MonitorElement* meClSizeXNotOnTrack_diskm1; 
    MonitorElement* meClSizeXNotOnTrack_diskm2;
    MonitorElement* meClSizeXNotOnTrack_diskm3;
    MonitorElement* meClSizeYOnTrack_all; 
    MonitorElement* meClSizeYOnTrack_bpix; 
    MonitorElement* meClSizeYOnTrack_fpix; 
    MonitorElement* meClSizeYOnTrack_layer1; 
    MonitorElement* meClSizeYOnTrack_layer2; 
    MonitorElement* meClSizeYOnTrack_layer3; 
    MonitorElement* meClSizeYOnTrack_layer4; 
    MonitorElement* meClSizeYOnTrack_diskp1; 
    MonitorElement* meClSizeYOnTrack_diskp2; 
    MonitorElement* meClSizeYOnTrack_diskp3; 
    MonitorElement* meClSizeYOnTrack_diskm1; 
    MonitorElement* meClSizeYOnTrack_diskm2; 
    MonitorElement* meClSizeYOnTrack_diskm3;
    MonitorElement* meClSizeYNotOnTrack_all; 
    MonitorElement* meClSizeYNotOnTrack_bpix; 
    MonitorElement* meClSizeYNotOnTrack_fpix; 
    MonitorElement* meClSizeYNotOnTrack_layer1; 
    MonitorElement* meClSizeYNotOnTrack_layer2; 
    MonitorElement* meClSizeYNotOnTrack_layer3; 
    MonitorElement* meClSizeYNotOnTrack_layer4; 
    MonitorElement* meClSizeYNotOnTrack_diskp1; 
    MonitorElement* meClSizeYNotOnTrack_diskp2; 
    MonitorElement* meClSizeYNotOnTrack_diskp3; 
    MonitorElement* meClSizeYNotOnTrack_diskm1; 
    MonitorElement* meClSizeYNotOnTrack_diskm2;
    MonitorElement* meClSizeYNotOnTrack_diskm3;

    //new
    MonitorElement* meNClustersOnTrack_all;
    MonitorElement* meNClustersOnTrack_bpix;
    MonitorElement* meNClustersOnTrack_fpix; 
    MonitorElement* meNClustersOnTrack_layer1; 
    MonitorElement* meNClustersOnTrack_layer2; 
    MonitorElement* meNClustersOnTrack_layer3; 
    MonitorElement* meNClustersOnTrack_layer4; 
    MonitorElement* meNClustersOnTrack_diskp1; 
    MonitorElement* meNClustersOnTrack_diskp2; 
    MonitorElement* meNClustersOnTrack_diskp3; 
    MonitorElement* meNClustersOnTrack_diskm1; 
    MonitorElement* meNClustersOnTrack_diskm2; 
    MonitorElement* meNClustersOnTrack_diskm3; 
    MonitorElement* meNClustersNotOnTrack_all; 
    MonitorElement* meNClustersNotOnTrack_bpix; 
    MonitorElement* meNClustersNotOnTrack_fpix; 
    MonitorElement* meNClustersNotOnTrack_layer1; 
    MonitorElement* meNClustersNotOnTrack_layer2; 
    MonitorElement* meNClustersNotOnTrack_layer3; 
    MonitorElement* meNClustersNotOnTrack_layer4; 
    MonitorElement* meNClustersNotOnTrack_diskp1; 
    MonitorElement* meNClustersNotOnTrack_diskp2; 
    MonitorElement* meNClustersNotOnTrack_diskp3; 
    MonitorElement* meNClustersNotOnTrack_diskm1; 
    MonitorElement* meNClustersNotOnTrack_diskm2;
    MonitorElement* meNClustersNotOnTrack_diskm3;
    //

    MonitorElement* meClPosLayer1OnTrack; 
    MonitorElement* meClPosLayer2OnTrack; 
    MonitorElement* meClPosLayer3OnTrack; 
    MonitorElement* meClPosLayer4OnTrack; 
    MonitorElement* meClPosLayer1NotOnTrack; 
    MonitorElement* meClPosLayer2NotOnTrack; 
    MonitorElement* meClPosLayer3NotOnTrack; 
    MonitorElement* meClPosLayer4NotOnTrack; 

    MonitorElement* meClPosDisk1pzOnTrack; 
    MonitorElement* meClPosDisk2pzOnTrack; 
    MonitorElement* meClPosDisk3pzOnTrack; 
    MonitorElement* meClPosDisk1mzOnTrack; 
    MonitorElement* meClPosDisk2mzOnTrack; 
    MonitorElement* meClPosDisk3mzOnTrack; 
    MonitorElement* meClPosDisk1pzNotOnTrack; 
    MonitorElement* meClPosDisk2pzNotOnTrack; 
    MonitorElement* meClPosDisk3pzNotOnTrack; 
    MonitorElement* meClPosDisk1mzNotOnTrack; 
    MonitorElement* meClPosDisk2mzNotOnTrack; 
    MonitorElement* meClPosDisk3mzNotOnTrack; 
    
    MonitorElement* meHitProbability;
};

#endif
