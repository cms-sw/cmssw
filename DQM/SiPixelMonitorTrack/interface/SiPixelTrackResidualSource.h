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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualModule.h"

//Files added for monitoring track quantities
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
//#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameWrapper.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

class SiPixelTrackResidualSource : public DQMEDAnalyzer {
  public:
    explicit SiPixelTrackResidualSource(const edm::ParameterSet&);
            ~SiPixelTrackResidualSource();

    virtual void dqmBeginRun(const edm::Run& r, edm::EventSetup const& iSetup);
    virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    void getrococcupancy(DetId detId,const edm::DetSetVector<PixelDigi> diginp,const TrackerTopology* const tTopo,std::vector<MonitorElement*> meinput);
    void triplets(double x1,double y1,double z1,double x2,double y2,double z2,double x3,double y3,double z3,
                  double ptsig, double & dc,double & dz, double kap); 

    std::string topFolderName_;

  private: 
    edm::ParameterSet pSet_; 
    edm::InputTag src_; 
    edm::InputTag clustersrc_; 
    edm::InputTag tracksrc_; 
    std::string ttrhbuilder_; 
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
    edm::EDGetTokenT<reco::TrackCollection> generalTracksToken_;
    edm::EDGetTokenT<std::vector<Trajectory> > tracksrcToken_;
    edm::EDGetTokenT<std::vector<reco::Track> > trackToken_;
    edm::EDGetTokenT<TrajTrackAssociationCollection> trackAssociationToken_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > clustersrcToken_;
    std::string vtxsrc_;
    edm::InputTag digisrc_;
    edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > digisrcToken_;

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

    std::vector<MonitorElement*> meResidualXSummedLay;
    std::vector<MonitorElement*> meResidualYSummedLay;
  
    MonitorElement* meNofTracks_;
    MonitorElement* meNofTracksInPixVol_;
    MonitorElement* meNofClustersOnTrack_;
    MonitorElement* meNofClustersNotOnTrack_;
    MonitorElement* meClChargeOnTrack_all; 
    MonitorElement* meClChargeOnTrack_bpix; 
    MonitorElement* meClChargeOnTrack_fpix; 
    std::vector<MonitorElement*> meClChargeOnTrack_layers;
    std::vector<MonitorElement*> meClChargeOnTrack_diskps;
    std::vector<MonitorElement*> meClChargeOnTrack_diskms;
    MonitorElement* meClChargeNotOnTrack_all; 
    MonitorElement* meClChargeNotOnTrack_bpix; 
    MonitorElement* meClChargeNotOnTrack_fpix; 
    std::vector<MonitorElement*> meClChargeNotOnTrack_layers;
    std::vector<MonitorElement*> meClChargeNotOnTrack_diskps;
    std::vector<MonitorElement*> meClChargeNotOnTrack_diskms;
    MonitorElement* meClSizeOnTrack_all; 
    MonitorElement* meClSizeOnTrack_bpix; 
    MonitorElement* meClSizeOnTrack_fpix; 
    std::vector<MonitorElement*> meClSizeOnTrack_layers;
    std::vector<MonitorElement*> meClSizeOnTrack_diskps;
    std::vector<MonitorElement*> meClSizeOnTrack_diskms;
    MonitorElement* meClSizeNotOnTrack_all; 
    MonitorElement* meClSizeNotOnTrack_bpix; 
    MonitorElement* meClSizeNotOnTrack_fpix; 
    std::vector<MonitorElement*> meClSizeNotOnTrack_layers;
    std::vector<MonitorElement*> meClSizeNotOnTrack_diskps;
    std::vector<MonitorElement*> meClSizeNotOnTrack_diskms;
    MonitorElement* meClSizeXOnTrack_all; 
    MonitorElement* meClSizeXOnTrack_bpix; 
    MonitorElement* meClSizeXOnTrack_fpix; 
    std::vector<MonitorElement*> meClSizeXOnTrack_layers;
    std::vector<MonitorElement*> meClSizeXOnTrack_diskps;
    std::vector<MonitorElement*> meClSizeXOnTrack_diskms;
    MonitorElement* meClSizeXNotOnTrack_all; 
    MonitorElement* meClSizeXNotOnTrack_bpix; 
    MonitorElement* meClSizeXNotOnTrack_fpix; 
    std::vector<MonitorElement*> meClSizeXNotOnTrack_layers;
    std::vector<MonitorElement*> meClSizeXNotOnTrack_diskps;
    std::vector<MonitorElement*> meClSizeXNotOnTrack_diskms;
    MonitorElement* meClSizeYOnTrack_all; 
    MonitorElement* meClSizeYOnTrack_bpix; 
    MonitorElement* meClSizeYOnTrack_fpix; 
    std::vector<MonitorElement*> meClSizeYOnTrack_layers;
    std::vector<MonitorElement*> meClSizeYOnTrack_diskps;
    std::vector<MonitorElement*> meClSizeYOnTrack_diskms;
    MonitorElement* meClSizeYNotOnTrack_all; 
    MonitorElement* meClSizeYNotOnTrack_bpix; 
    MonitorElement* meClSizeYNotOnTrack_fpix; 
    std::vector<MonitorElement*> meClSizeYNotOnTrack_layers;
    std::vector<MonitorElement*> meClSizeYNotOnTrack_diskps;
    std::vector<MonitorElement*> meClSizeYNotOnTrack_diskms;

    //new
    MonitorElement* meNClustersOnTrack_all;
    MonitorElement* meNClustersOnTrack_bpix;
    MonitorElement* meNClustersOnTrack_fpix; 
    std::vector<MonitorElement*> meNClustersOnTrack_layers;
    std::vector<MonitorElement*> meNClustersOnTrack_diskps;
    std::vector<MonitorElement*> meNClustersOnTrack_diskms;
    MonitorElement* meNClustersNotOnTrack_all; 
    MonitorElement* meNClustersNotOnTrack_bpix; 
    MonitorElement* meNClustersNotOnTrack_fpix; 
    std::vector<MonitorElement*> meNClustersNotOnTrack_layers;
    std::vector<MonitorElement*> meNClustersNotOnTrack_diskps;
    std::vector<MonitorElement*> meNClustersNotOnTrack_diskms;
    //

    std::vector<MonitorElement*> meClPosLayersOnTrack;
    std::vector<MonitorElement*> meClPosLayersLadVsModOnTrack;
    std::vector<MonitorElement*> meClPosLayersNotOnTrack;
    std::vector<MonitorElement*> meClPosDiskspzOnTrack;
    std::vector<MonitorElement*> meClPosDisksmzOnTrack;
    std::vector<MonitorElement*> meClPosDiskspzNotOnTrack;
    std::vector<MonitorElement*> meClPosDisksmzNotOnTrack;

    std::vector<MonitorElement*> meZeroRocLadvsModOnTrackBarrel;
    std::vector<MonitorElement*> meZeroRocLadvsModOffTrackBarrel;
    
    MonitorElement* meHitProbability;
    
    int noOfLayers;
    int noOfDisks;
};

#endif
