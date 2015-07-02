// -*- C++ -*-
//
// Package:    DPGAnalysis
// Class:      TrackerDpgAnalysis
//
/**\class TrackerDpgAnalysis TrackerDpgAnalysis.cc DPGAnalysis/SiStripTools/plugins/TrackerDpgAnalysis.cc

 Description: analysis of the clusters and digis in the tracker

 Implementation:
      analysis of the clusters and digis in the tracker
*/
//
// Original Author:  Christophe DELAERE
//         Created:  Tue Sep 23 02:11:44 CEST 2008
//         Revised:  Thu Nov 26 10:00:00 CEST 2009
// part of the code was inspired by http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/YGao/LhcTrackAnalyzer/
// part of the code was inspired by
// other inputs from Andrea Giammanco, Gaelle Boudoul, Andrea Venturi, Steven Lowette, Gavril Giurgiu
// $Id: TrackerDpgAnalysis.cc,v 1.14 2013/02/27 19:49:47 wmtan Exp $
//
//

// system include files
#include <memory>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>
#include <string.h>
#include <sstream>
#include <fstream>

// root include files
#include "TTree.h"
#include "TFile.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include <CondFormats/SiStripObjects/interface/FedChannelConnection.h>
#include <CondFormats/SiStripObjects/interface/SiStripFedCabling.h>
#include <CondFormats/DataRecord/interface/SiStripFedCablingRcd.h>
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include <Geometry/CommonTopologies/interface/Topology.h>
#include <Geometry/CommonTopologies/interface/StripTopology.h>
#include <Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h>
#include <Geometry/CommonTopologies/interface/PixelTopology.h>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include <DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h>
#include <DataFormats/SiStripDetId/interface/SiStripDetId.h>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include <DataFormats/SiPixelCluster/interface/SiPixelCluster.h>
#include <DataFormats/TrackReco/interface/TrackFwd.h>
#include <DataFormats/TrackReco/interface/Track.h>
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include <DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h>
#include <DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h>
#include <DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h>
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include <DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h>
#include <DataFormats/Common/interface/TriggerResults.h>
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include <MagneticField/Engine/interface/MagneticField.h>
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>
#include <RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h>
#include <RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h>
#include <RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit1D.h>
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include <TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h>
#include <TrackingTools/PatternTools/interface/Trajectory.h>
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include <RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h>

// topology
#include "DPGAnalysis/SiStripTools/interface/EventShape.h"

//
// class decleration
//
typedef math::XYZPoint Point;

class TrackerDpgAnalysis : public edm::EDAnalyzer {
   public:
      explicit TrackerDpgAnalysis(const edm::ParameterSet&);
      ~TrackerDpgAnalysis();

   protected:
      std::vector<double> onTrackAngles(edm::Handle<edmNew::DetSetVector<SiStripCluster> >&,const std::vector<Trajectory>& );
      void insertMeasurement(std::multimap<const uint32_t,std::pair<LocalPoint,double> >&, const TransientTrackingRecHit*,double);
      std::vector<int> onTrack(edm::Handle<edmNew::DetSetVector<SiStripCluster> >&,const reco::TrackCollection&, uint32_t );
      void insertMeasurement(std::multimap<const uint32_t,std::pair<int,int> >&, const TrackingRecHit*,int);
      std::map<size_t,int> inVertex(const reco::TrackCollection&, const reco::VertexCollection&, uint32_t);
      std::vector<std::pair<double,double> > onTrackAngles(edm::Handle<edmNew::DetSetVector<SiPixelCluster> >&,const std::vector<Trajectory>& );
      void insertMeasurement(std::multimap<const uint32_t,std::pair<LocalPoint,std::pair<double,double> > >&, const TransientTrackingRecHit*,double,double);
      std::vector<int> onTrack(edm::Handle<edmNew::DetSetVector<SiPixelCluster> >&,const reco::TrackCollection&, uint32_t );
      void insertMeasurement(std::multimap<const uint32_t,std::pair<std::pair<float, float>,int> >&, const TrackingRecHit*,int);
      std::string toStringName(uint32_t, const TrackerTopology*);
      std::string toStringId(uint32_t);
      double sumPtSquared(const reco::Vertex&);
      float delay(const SiStripEventSummary&);
      std::map<uint32_t,float> delay(const std::vector<std::string>&);

   private:
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------
      static const int nMaxPVs_ = 50;
      edm::EDGetTokenT<SiStripEventSummary> summaryToken_;
      edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterToken_;
      edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelclusterToken_;
      edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > dedx1Token_;
      edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > dedx2Token_;
      edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > dedx3Token_;
      edm::EDGetTokenT<reco::VertexCollection> pixelVertexToken_;
      edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
      edm::EDGetTokenT<reco::BeamSpot> bsToken_;
      edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> L1Token_;
      edm::InputTag HLTTag_;
      edm::EDGetTokenT<edm::TriggerResults> HLTToken_;
      std::vector<edm::InputTag> trackLabels_;
      std::vector<edm::EDGetTokenT<reco::TrackCollection> > trackTokens_;
      std::vector<edm::EDGetTokenT<std::vector<Trajectory> > > trajectoryTokens_;
      std::vector<edm::EDGetTokenT<TrajTrackAssociationCollection> > trajTrackAssoTokens_;
      edm::ESHandle<SiStripFedCabling> cabling_;
      edm::ESHandle<TrackerGeometry> tracker_;
      std::multimap<const uint32_t,const FedChannelConnection*> connections_;
      bool functionality_offtrackClusters_, functionality_ontrackClusters_, functionality_pixclusters_, functionality_pixvertices_,
           functionality_missingHits_, functionality_tracks_, functionality_vertices_, functionality_events_;
      TTree* clusters_;
      TTree* pixclusters_;
      std::vector<TTree*> tracks_;
      std::vector<TTree*> missingHits_;
      TTree* vertices_;
      TTree* pixelVertices_;
      TTree* event_;
      TTree* psumap_;
      TTree* readoutmap_;
      bool onTrack_;
      uint32_t vertexid_;
      edm::EventNumber_t eventid_;
      uint32_t runid_;
      uint32_t globalvertexid_;
      uint32_t *globaltrackid_, *trackid_;
      float globalX_, globalY_, globalZ_;
      float measX_, measY_, errorX_, errorY_;
      float angle_, maxCharge_;
      float clCorrectedCharge_, clCorrectedSignalOverNoise_;
      float clNormalizedCharge_, clNormalizedNoise_, clSignalOverNoise_;
      float clBareNoise_, clBareCharge_;
      float clWidth_, clPosition_, thickness_, stripLength_, distance_;
      float eta_, phi_, chi2_;
      float dedx1_, dedx2_, dedx3_;
      uint32_t detid_, dcuId_, type_;
      uint16_t fecCrate_, fecSlot_, fecRing_, ccuAdd_, ccuChan_, lldChannel_, fedId_, fedCh_, fiberLength_;
      uint32_t nclusters_, npixClusters_, nclustersOntrack_, npixClustersOntrack_, dedxNoM_, quality_, foundhits_, lostHits_, ndof_;
      uint32_t *ntracks_, *ntrajs_;
      float *lowPixelProbabilityFraction_;
      uint32_t nVertices_, nPixelVertices_, nLayers_,foundhitsStrips_,foundhitsPixels_,losthitsStrips_,losthitsPixels_;
      uint32_t nTracks_pvtx_;
      uint32_t clSize_, clSizeX_, clSizeY_;
      float fBz_, clPositionX_, clPositionY_, alpha_, beta_, chargeCorr_;
      float recx_pvtx_, recy_pvtx_, recz_pvtx_,
            recx_err_pvtx_, recy_err_pvtx_, recz_err_pvtx_, sumptsq_pvtx_;
      float pterr_, etaerr_, phierr_;
      float dz_, dzerr_, dzCorr_, dxy_, dxyerr_, dxyCorr_;
      float qoverp_, xPCA_, yPCA_, zPCA_, trkWeightpvtx_;
      bool isValid_pvtx_, isFake_pvtx_;
      float charge_, p_, pt_;
      float bsX0_, bsY0_, bsZ0_, bsSigmaZ_, bsDxdz_, bsDydz_;
      float thrustValue_, thrustX_, thrustY_, thrustZ_, sphericity_, planarity_, aplanarity_, delay_;
      bool L1DecisionBits_[192], L1TechnicalBits_[64], HLTDecisionBits_[256];
      uint32_t orbit_, orbitL1_, bx_, store_, time_;
      uint16_t lumiSegment_, physicsDeclared_;
      char *moduleName_, *moduleId_, *PSUname_;
      std::string cablingFileName_;
      std::vector<std::string> delayFileNames_;
      edm::ParameterSet pset_;
      std::vector<std::string>  hlNames_;  // name of each HLT algorithm
      HLTConfigProvider hltConfig_;        // to get configuration for L1s/Pre

};

//
// constructors and destructor
//
TrackerDpgAnalysis::TrackerDpgAnalysis(const edm::ParameterSet& iConfig):hltConfig_()
{
   // members
   moduleName_ = new char[256];
   moduleId_   = new char[256];
   PSUname_    = new char[256];
   pset_ = iConfig;

   // enable/disable functionalities
   functionality_offtrackClusters_ = iConfig.getUntrackedParameter<bool>("keepOfftrackClusters",true);
   functionality_ontrackClusters_  = iConfig.getUntrackedParameter<bool>("keepOntrackClusters",true);
   functionality_pixclusters_      = iConfig.getUntrackedParameter<bool>("keepPixelClusters",true);
   functionality_pixvertices_      = iConfig.getUntrackedParameter<bool>("keepPixelVertices",true);
   functionality_missingHits_      = iConfig.getUntrackedParameter<bool>("keepMissingHits",true);
   functionality_tracks_           = iConfig.getUntrackedParameter<bool>("keepTracks",true);
   functionality_vertices_         = iConfig.getUntrackedParameter<bool>("keepVertices",true);
   functionality_events_           = iConfig.getUntrackedParameter<bool>("keepEvents",true);

   // parameters
   summaryToken_      = consumes<SiStripEventSummary>(edm::InputTag("siStripDigis"));
   clusterToken_      = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("ClustersLabel"));
   pixelclusterToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("PixelClustersLabel"));
   trackLabels_       = iConfig.getParameter<std::vector<edm::InputTag> >("TracksLabel");
   trackTokens_       = edm::vector_transform(trackLabels_, [this](edm::InputTag const & tag){return consumes<reco::TrackCollection>(tag);});
   trajectoryTokens_  = edm::vector_transform(trackLabels_, [this](edm::InputTag const & tag){return consumes<std::vector<Trajectory> >(tag);});
   trajTrackAssoTokens_ = edm::vector_transform(trackLabels_, [this](edm::InputTag const & tag){return consumes<TrajTrackAssociationCollection>(tag);});
   dedx1Token_        = consumes<edm::ValueMap<reco::DeDxData> >(iConfig.getParameter<edm::InputTag>("DeDx1Label"));
   dedx2Token_        = consumes<edm::ValueMap<reco::DeDxData> >(iConfig.getParameter<edm::InputTag>("DeDx2Label"));
   dedx3Token_        = consumes<edm::ValueMap<reco::DeDxData> >(iConfig.getParameter<edm::InputTag>("DeDx3Label"));
   vertexToken_       = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexLabel"));
   pixelVertexToken_  = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("pixelVertexLabel"));
   bsToken_           = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotLabel"));
   L1Token_           = consumes<L1GlobalTriggerReadoutRecord>(iConfig.getParameter<edm::InputTag>("L1Label"));
   HLTTag_            = iConfig.getParameter<edm::InputTag>("HLTLabel");
   HLTToken_          = consumes<edm::TriggerResults>(HLTTag_);

   // initialize arrays
   size_t trackSize(trackLabels_.size());
   ntracks_ = new uint32_t[trackSize];
   ntrajs_ = new uint32_t[trackSize];
   globaltrackid_  = new uint32_t[trackSize];
   trackid_ = new uint32_t[trackSize];
   lowPixelProbabilityFraction_ = new float[trackSize];
   globalvertexid_ = iConfig.getParameter<uint32_t>("InitalCounter");
   for(size_t i = 0; i<trackSize;++i) {
     ntracks_[i]=0;
     ntrajs_[i]=0;
     globaltrackid_[i]=iConfig.getParameter<uint32_t>("InitalCounter");
     trackid_[i]=0;
     lowPixelProbabilityFraction_[i]=0;
   }

   // create output
   edm::Service<TFileService> fileService;
   TFileDirectory* dir = new TFileDirectory(fileService->mkdir("trackerDPG"));

   // create a TTree for clusters
   clusters_ = dir->make<TTree>("clusters","cluster information");
   clusters_->Branch("eventid",&eventid_,"eventid/i");
   clusters_->Branch("runid",&runid_,"runid/i");
   for(size_t i = 0; i<trackSize; ++i) {
     char buffer1[256];
     char buffer2[256];
     sprintf(buffer1,"trackid%lu",(unsigned long)i);
     sprintf(buffer2,"trackid%lu/i",(unsigned long)i);
     clusters_->Branch(buffer1,trackid_+i,buffer2);
   }
   clusters_->Branch("onTrack",&onTrack_,"onTrack/O");
   clusters_->Branch("clWidth",&clWidth_,"clWidth/F");
   clusters_->Branch("clPosition",&clPosition_,"clPosition/F");
   clusters_->Branch("clglobalX",&globalX_,"clglobalX/F");
   clusters_->Branch("clglobalY",&globalY_,"clglobalY/F");
   clusters_->Branch("clglobalZ",&globalZ_,"clglobalZ/F");
   clusters_->Branch("angle",&angle_,"angle/F");
   clusters_->Branch("thickness",&thickness_,"thickness/F");
   clusters_->Branch("maxCharge",&maxCharge_,"maxCharge/F");
   clusters_->Branch("clNormalizedCharge",&clNormalizedCharge_,"clNormalizedCharge/F");
   clusters_->Branch("clNormalizedNoise",&clNormalizedNoise_,"clNormalizedNoise/F");
   clusters_->Branch("clSignalOverNoise",&clSignalOverNoise_,"clSignalOverNoise/F");
   clusters_->Branch("clCorrectedCharge",&clCorrectedCharge_,"clCorrectedCharge/F");
   clusters_->Branch("clCorrectedSignalOverNoise",&clCorrectedSignalOverNoise_,"clCorrectedSignalOverNoise/F");
   clusters_->Branch("clBareCharge",&clBareCharge_,"clBareCharge/F");
   clusters_->Branch("clBareNoise",&clBareNoise_,"clBareNoise/F");
   clusters_->Branch("stripLength",&stripLength_,"stripLength/F");
   clusters_->Branch("detid",&detid_,"detid/i");
   clusters_->Branch("lldChannel",&lldChannel_,"lldChannel/s");

   // create a TTree for pixel clusters
   pixclusters_ = dir->make<TTree>("pixclusters","pixel cluster information");
   pixclusters_->Branch("eventid",&eventid_,"eventid/i");
   pixclusters_->Branch("runid",&runid_,"runid/i");
   for(size_t i = 0; i<trackSize; ++i) {
     char buffer1[256];
     char buffer2[256];
     sprintf(buffer1,"trackid%lu",(unsigned long)i);
     sprintf(buffer2,"trackid%lu/i",(unsigned long)i);
     pixclusters_->Branch(buffer1,trackid_+i,buffer2);
   }
   pixclusters_->Branch("onTrack",&onTrack_,"onTrack/O");
   pixclusters_->Branch("clPositionX",&clPositionX_,"clPositionX/F");
   pixclusters_->Branch("clPositionY",&clPositionY_,"clPositionY/F");
   pixclusters_->Branch("clSize",&clSize_,"clSize/i");
   pixclusters_->Branch("clSizeX",&clSizeX_,"clSizeX/i");
   pixclusters_->Branch("clSizeY",&clSizeY_,"clSizeY/i");
   pixclusters_->Branch("alpha",&alpha_,"alpha/F");
   pixclusters_->Branch("beta",&beta_,"beta/F");
   pixclusters_->Branch("charge",&charge_,"charge/F");
   pixclusters_->Branch("chargeCorr",&chargeCorr_,"chargeCorr/F");
   pixclusters_->Branch("clglobalX",&globalX_,"clglobalX/F");
   pixclusters_->Branch("clglobalY",&globalY_,"clglobalY/F");
   pixclusters_->Branch("clglobalZ",&globalZ_,"clglobalZ/F");
   pixclusters_->Branch("detid",&detid_,"detid/i");

   // create a tree for tracks
   for(size_t i = 0; i<trackSize; ++i) {
     char buffer1[256];
     char buffer2[256];
     sprintf(buffer1,"tracks%lu",(unsigned long)i);
     sprintf(buffer2,"track%lu information",(unsigned long)i);
     TTree* thetracks_ = dir->make<TTree>(buffer1,buffer2);
     sprintf(buffer1,"trackid%lu",(unsigned long)i);
     sprintf(buffer2,"trackid%lu/i",(unsigned long)i);
     thetracks_->Branch(buffer1,globaltrackid_+i,buffer2);
     thetracks_->Branch("eventid",&eventid_,"eventid/i");
     thetracks_->Branch("runid",&runid_,"runid/i");
     thetracks_->Branch("chi2",&chi2_,"chi2/F");
     thetracks_->Branch("eta",&eta_,"eta/F");
     thetracks_->Branch("etaerr",&etaerr_,"etaerr/F");
     thetracks_->Branch("phi",&phi_,"phi/F");
     thetracks_->Branch("phierr",&phierr_,"phierr/F");
     thetracks_->Branch("dedx1",&dedx1_,"dedx1/F");
     thetracks_->Branch("dedx2",&dedx2_,"dedx2/F");
     thetracks_->Branch("dedx3",&dedx3_,"dedx3/F");
     thetracks_->Branch("dedxNoM",&dedxNoM_,"dedxNoM/i");
     thetracks_->Branch("charge",&charge_,"charge/F");
     thetracks_->Branch("quality",&quality_,"quality/i");
     thetracks_->Branch("foundhits",&foundhits_,"foundhits/i");
     thetracks_->Branch("lostHits",&lostHits_,"lostHits/i");
     thetracks_->Branch("foundhitsStrips",&foundhitsStrips_,"foundhitsStrips/i");
     thetracks_->Branch("foundhitsPixels",&foundhitsPixels_,"foundhitsPixels/i");
     thetracks_->Branch("losthitsStrips",&losthitsStrips_,"losthitsStrips/i");
     thetracks_->Branch("losthitsPixels",&losthitsPixels_,"losthitsPixels/i");
     thetracks_->Branch("p",&p_,"p/F");
     thetracks_->Branch("pt",&pt_,"pt/F");
     thetracks_->Branch("pterr",&pterr_,"pterr/F");
     thetracks_->Branch("ndof",&ndof_,"ndof/i");
     thetracks_->Branch("dz",&dz_,"dz/F");
     thetracks_->Branch("dzerr",&dzerr_,"dzerr/F");
     thetracks_->Branch("dzCorr",&dzCorr_,"dzCorr/F");
     thetracks_->Branch("dxy",&dxy_,"dxy/F");
     thetracks_->Branch("dxyerr",&dxyerr_,"dxyerr/F");
     thetracks_->Branch("dxyCorr",&dxyCorr_,"dxyCorr/F");
     thetracks_->Branch("qoverp",&qoverp_,"qoverp/F");
     thetracks_->Branch("xPCA",&xPCA_,"xPCA/F");
     thetracks_->Branch("yPCA",&yPCA_,"yPCA/F");
     thetracks_->Branch("zPCA",&zPCA_,"zPCA/F");
     thetracks_->Branch("nLayers",&nLayers_,"nLayers/i");
     thetracks_->Branch("trkWeightpvtx",&trkWeightpvtx_,"trkWeightpvtx/F");
     thetracks_->Branch("vertexid",&vertexid_,"vertexid/i");
     tracks_.push_back(thetracks_);
   }

   // create a tree for missing hits
   for(size_t i = 0; i<trackSize; ++i) {
     char buffer1[256];
     char buffer2[256];
     sprintf(buffer1,"misingHits%lu",(unsigned long)i);
     sprintf(buffer2,"missing hits from track collection %lu",(unsigned long)i);
     TTree* themissingHits_ = dir->make<TTree>(buffer1,buffer2);
     sprintf(buffer1,"trackid%lu",(unsigned long)i);
     sprintf(buffer2,"trackid%lu/i",(unsigned long)i);
     themissingHits_->Branch(buffer1,globaltrackid_+i,buffer2);
     themissingHits_->Branch("eventid",&eventid_,"eventid/i");
     themissingHits_->Branch("runid",&runid_,"runid/i");
     themissingHits_->Branch("detid",&detid_,"detid/i");
     themissingHits_->Branch("type",&type_,"type/i");
     themissingHits_->Branch("localX",&clPositionX_,"localX/F");
     themissingHits_->Branch("localY",&clPositionY_,"localY/F");
     themissingHits_->Branch("globalX",&globalX_,"globalX/F");
     themissingHits_->Branch("globalY",&globalY_,"globalY/F");
     themissingHits_->Branch("globalZ",&globalZ_,"globalZ/F");
     themissingHits_->Branch("measX",&measX_,"measX/F");
     themissingHits_->Branch("measY",&measY_,"measY/F");
     themissingHits_->Branch("errorX",&errorX_,"errorX/F");
     themissingHits_->Branch("errorY",&errorY_,"errorY/F");
     missingHits_.push_back(themissingHits_);
   }

   // create a tree for the vertices
   vertices_ = dir->make<TTree>("vertices","vertex information");
   vertices_->Branch("vertexid",&globalvertexid_,"vertexid/i");
   vertices_->Branch("eventid",&eventid_,"eventid/i");
   vertices_->Branch("runid",&runid_,"runid/i");
   vertices_->Branch("nTracks",&nTracks_pvtx_,"nTracks/i");
   vertices_->Branch("sumptsq",&sumptsq_pvtx_,"sumptsq/F");
   vertices_->Branch("isValid",&isValid_pvtx_,"isValid/O");
   vertices_->Branch("isFake",&isFake_pvtx_,"isFake/O");
   vertices_->Branch("recx",&recx_pvtx_,"recx/F");
   vertices_->Branch("recy",&recy_pvtx_,"recy/F");
   vertices_->Branch("recz",&recz_pvtx_,"recz/F");
   vertices_->Branch("recx_err",&recx_err_pvtx_,"recx_err/F");
   vertices_->Branch("recy_err",&recy_err_pvtx_,"recy_err/F");
   vertices_->Branch("recz_err",&recz_err_pvtx_,"recz_err/F");

   // create a tree for the vertices
   pixelVertices_ = dir->make<TTree>("pixelVertices","pixel vertex information");
   pixelVertices_->Branch("eventid",&eventid_,"eventid/i");
   pixelVertices_->Branch("runid",&runid_,"runid/i");
   pixelVertices_->Branch("nTracks",&nTracks_pvtx_,"nTracks/i");
   pixelVertices_->Branch("sumptsq",&sumptsq_pvtx_,"sumptsq/F");
   pixelVertices_->Branch("isValid",&isValid_pvtx_,"isValid/O");
   pixelVertices_->Branch("isFake",&isFake_pvtx_,"isFake/O");
   pixelVertices_->Branch("recx",&recx_pvtx_,"recx/F");
   pixelVertices_->Branch("recy",&recy_pvtx_,"recy/F");
   pixelVertices_->Branch("recz",&recz_pvtx_,"recz/F");
   pixelVertices_->Branch("recx_err",&recx_err_pvtx_,"recx_err/F");
   pixelVertices_->Branch("recy_err",&recy_err_pvtx_,"recy_err/F");
   pixelVertices_->Branch("recz_err",&recz_err_pvtx_,"recz_err/F");

   // create a tree for the events
   event_ = dir->make<TTree>("events","event information");
   event_->Branch("eventid",&eventid_,"eventid/i");
   event_->Branch("runid",&runid_,"runid/i");
   event_->Branch("L1DecisionBits",L1DecisionBits_,"L1DecisionBits[192]/O");
   event_->Branch("L1TechnicalBits",L1TechnicalBits_,"L1TechnicalBits[64]/O");
   event_->Branch("orbit",&orbit_,"orbit/i");
   event_->Branch("orbitL1",&orbitL1_,"orbitL1/i");
   event_->Branch("bx",&bx_,"bx/i");
   event_->Branch("store",&store_,"store/i");
   event_->Branch("time",&time_,"time/i");
   event_->Branch("delay",&delay_,"delay/F");
   event_->Branch("lumiSegment",&lumiSegment_,"lumiSegment/s");
   event_->Branch("physicsDeclared",&physicsDeclared_,"physicsDeclared/s");
   event_->Branch("HLTDecisionBits",HLTDecisionBits_,"HLTDecisionBits[256]/O");
   char buffer[256];
   sprintf(buffer,"ntracks[%lu]/i",(unsigned long)trackSize);
   event_->Branch("ntracks",ntracks_,buffer);
   sprintf(buffer,"ntrajs[%lu]/i",(unsigned long)trackSize);
   event_->Branch("ntrajs",ntrajs_,buffer);
   sprintf(buffer,"lowPixelProbabilityFraction[%lu]/F",(unsigned long)trackSize);
   event_->Branch("lowPixelProbabilityFraction",lowPixelProbabilityFraction_,buffer);
   event_->Branch("nclusters",&nclusters_,"nclusters/i");
   event_->Branch("npixClusters",&npixClusters_,"npixClusters/i");
   event_->Branch("nclustersOntrack",&nclustersOntrack_,"nclustersOntrack/i");
   event_->Branch("npixClustersOntrack",&npixClustersOntrack_,"npixClustersOntrack/i");
   event_->Branch("bsX0",&bsX0_,"bsX0/F");
   event_->Branch("bsY0",&bsY0_,"bsY0/F");
   event_->Branch("bsZ0",&bsZ0_,"bsZ0/F");
   event_->Branch("bsSigmaZ",&bsSigmaZ_,"bsSigmaZ/F");
   event_->Branch("bsDxdz",&bsDxdz_,"bsDxdz/F");
   event_->Branch("bsDydz",&bsDydz_,"bsDydz/F");
   event_->Branch("nVertices",&nVertices_,"nVertices/i");
   event_->Branch("thrustValue",&thrustValue_,"thrustValue/F");
   event_->Branch("thrustX",&thrustX_,"thrustX/F");
   event_->Branch("thrustY",&thrustY_,"thrustY/F");
   event_->Branch("thrustZ",&thrustZ_,"thrustZ/F");
   event_->Branch("sphericity",&sphericity_,"sphericity/F");
   event_->Branch("planarity",&planarity_,"planarity/F");
   event_->Branch("aplanarity",&aplanarity_,"aplanarity/F");
   event_->Branch("MagneticField",&fBz_,"MagneticField/F");

   // cabling
   cablingFileName_ = iConfig.getUntrackedParameter<std::string>("PSUFileName","PSUmapping.csv");
   delayFileNames_  = iConfig.getUntrackedParameter<std::vector<std::string> >("DelayFileNames",std::vector<std::string>(0));
   psumap_ = dir->make<TTree>("psumap","PSU map");
   psumap_->Branch("PSUname",PSUname_,"PSUname/C");
   psumap_->Branch("dcuId",&dcuId_,"dcuId/i");
   readoutmap_ = dir->make<TTree>("readoutMap","cabling map");
   readoutmap_->Branch("detid",&detid_,"detid/i");
   readoutmap_->Branch("dcuId",&dcuId_,"dcuId/i");
   readoutmap_->Branch("fecCrate",&fecCrate_,"fecCrate/s");
   readoutmap_->Branch("fecSlot",&fecSlot_,"fecSlot/s");
   readoutmap_->Branch("fecRing",&fecRing_,"fecRing/s");
   readoutmap_->Branch("ccuAdd",&ccuAdd_,"ccuAdd/s");
   readoutmap_->Branch("ccuChan",&ccuChan_,"ccuChan/s");
   readoutmap_->Branch("lldChannel",&lldChannel_,"lldChannel/s");
   readoutmap_->Branch("fedId",&fedId_,"fedId/s");
   readoutmap_->Branch("fedCh",&fedCh_,"fedCh/s");
   readoutmap_->Branch("fiberLength",&fiberLength_,"fiberLength/s");
   readoutmap_->Branch("moduleName",moduleName_,"moduleName/C");
   readoutmap_->Branch("moduleId",moduleId_,"moduleId/C");
   readoutmap_->Branch("delay",&delay_,"delay/F");
   readoutmap_->Branch("globalX",&globalX_,"globalX/F");
   readoutmap_->Branch("globalY",&globalY_,"globalY/F");
   readoutmap_->Branch("globalZ",&globalZ_,"globalZ/F");
}

TrackerDpgAnalysis::~TrackerDpgAnalysis()
{
  delete[] moduleName_;
  delete[] moduleId_;
}

//
// member functions
//

// ------------ method called to for each event  ------------
void
TrackerDpgAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;
   using reco::TrackCollection;

   // load event info
   eventid_     = iEvent.id().event();
   runid_       = iEvent.id().run();
   bx_          = iEvent.eventAuxiliary().bunchCrossing();
   orbit_       = iEvent.eventAuxiliary().orbitNumber();
   store_       = iEvent.eventAuxiliary().storeNumber();
   time_        = iEvent.eventAuxiliary().time().value();
   lumiSegment_ = iEvent.eventAuxiliary().luminosityBlock();

   // Retrieve commissioning information from "event summary", when available (for standard fine delay)
   edm::Handle<SiStripEventSummary> summary;
   iEvent.getByToken(summaryToken_, summary );
   if(summary.isValid())
     delay_ = delay(*summary.product());
   else
     delay_ = 0.;

   // -- Magnetic field
   ESHandle<MagneticField> MF;
   iSetup.get<IdealMagneticFieldRecord>().get(MF);
   const MagneticField* theMagneticField = MF.product();
   fBz_   = fabs(theMagneticField->inTesla(GlobalPoint(0,0,0)).z());

   // load trigger info
   edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
   iEvent.getByToken(L1Token_, gtrr_handle);
   L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();
   L1GtFdlWord fdlWord = gtrr->gtFdlWord();
   DecisionWord L1decision = fdlWord.gtDecisionWord();
   for(int bit=0;bit<128;++bit) {
     L1DecisionBits_[bit] = L1decision[bit];
   }
   DecisionWordExtended L1decisionE = fdlWord.gtDecisionWordExtended();
   for(int bit=0;bit<64;++bit) {
     L1DecisionBits_[bit+128] = L1decisionE[bit];
   }
   TechnicalTriggerWord L1technical = fdlWord.gtTechnicalTriggerWord();
   for(int bit=0;bit<64;++bit) {
     L1TechnicalBits_[bit] = L1technical[bit];
   }
   orbitL1_ = fdlWord.orbitNr();
   physicsDeclared_ = fdlWord.physicsDeclared();
   edm::Handle<edm::TriggerResults> trh;
   iEvent.getByToken(HLTToken_, trh);
   size_t ntrh = trh->size();
   for(size_t bit=0;bit<256;++bit)
     HLTDecisionBits_[bit] = bit<ntrh ? (bool)(trh->accept(bit)): false;

   // load beamspot
   edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
   iEvent.getByToken(bsToken_,recoBeamSpotHandle);
   reco::BeamSpot bs = *recoBeamSpotHandle;
   const Point beamSpot = recoBeamSpotHandle.isValid() ?
                           Point(recoBeamSpotHandle->x0(), recoBeamSpotHandle->y0(), recoBeamSpotHandle->z0()) :
			   Point(0, 0, 0);
   if(recoBeamSpotHandle.isValid()) {
     bsX0_ = bs.x0();
     bsY0_ = bs.y0();
     bsZ0_ = bs.z0();
     bsSigmaZ_ = bs.sigmaZ();
     bsDxdz_ = bs.dxdz();
     bsDydz_ = bs.dydz();
   } else {
     bsX0_ = 0.;
     bsY0_ = 0.;
     bsZ0_ = 0.;
     bsSigmaZ_ = 0.;
     bsDxdz_ = 0.;
     bsDydz_ = 0.;
   }

   // load primary vertex
   static const reco::VertexCollection s_empty_vertexColl;
   edm::Handle<reco::VertexCollection> vertexCollectionHandle;
   iEvent.getByToken(vertexToken_,vertexCollectionHandle);
   const reco::VertexCollection vertexColl = *(vertexCollectionHandle.product());
   nVertices_ = 0;
   for(reco::VertexCollection::const_iterator v=vertexColl.begin();
       v!=vertexColl.end(); ++v) {
     if(v->isValid() && !v->isFake()) ++nVertices_;
   }

   // load pixel vertices
   // Pixel vertices are handled as primary vertices, but not linked to tracks.
    edm::Handle<reco::VertexCollection>  pixelVertexCollectionHandle;
    iEvent.getByToken(pixelVertexToken_, pixelVertexCollectionHandle);
    const reco::VertexCollection pixelVertexColl = *(pixelVertexCollectionHandle.product());
    nPixelVertices_ = pixelVertexColl.size();

   // load the clusters
   edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
   iEvent.getByToken(clusterToken_,clusters);
   edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelclusters;
   iEvent.getByToken(pixelclusterToken_,pixelclusters  );

   // load dedx info
   Handle<ValueMap<DeDxData> > dEdx1Handle;
   Handle<ValueMap<DeDxData> > dEdx2Handle;
   Handle<ValueMap<DeDxData> > dEdx3Handle;
   try {iEvent.getByToken(dedx1Token_, dEdx1Handle);} catch ( cms::Exception& ) {;}
   try {iEvent.getByToken(dedx2Token_, dEdx2Handle);} catch ( cms::Exception& ) {;}
   try {iEvent.getByToken(dedx3Token_, dEdx3Handle);} catch ( cms::Exception& ) {;}
   const ValueMap<DeDxData> dEdxTrack1 = *dEdx1Handle.product();
   const ValueMap<DeDxData> dEdxTrack2 = *dEdx2Handle.product();
   const ValueMap<DeDxData> dEdxTrack3 = *dEdx3Handle.product();

   // load track collections
   const size_t trackSize(trackLabels_.size());
   std::vector<reco::TrackCollection> trackCollection;
   std::vector<edm::Handle<reco::TrackCollection> > trackCollectionHandle;
   trackCollectionHandle.resize(trackSize);
   size_t index = 0;
   for(std::vector<edm::EDGetTokenT<reco::TrackCollection> >::const_iterator token = trackTokens_.begin();token!=trackTokens_.end();++token,++index) {
     try {iEvent.getByToken(*token,trackCollectionHandle[index]);} catch ( cms::Exception& ) {;}
     trackCollection.push_back(*trackCollectionHandle[index].product());
     ntracks_[index] = trackCollection[index].size();
   }

   // load the trajectory collections
   std::vector<std::vector<Trajectory> > trajectoryCollection;
   std::vector<edm::Handle<std::vector<Trajectory> > > trajectoryCollectionHandle;
   trajectoryCollectionHandle.resize(trackSize);
   index = 0;
   for(std::vector<edm::EDGetTokenT<std::vector<Trajectory> > >::const_iterator token = trajectoryTokens_.begin();token!=trajectoryTokens_.end();++token,++index) {
     try {iEvent.getByToken(*token,trajectoryCollectionHandle[index]);} catch ( cms::Exception& ) {;}
     trajectoryCollection.push_back(*trajectoryCollectionHandle[index].product());
     ntrajs_[index] = trajectoryCollection[index].size();
   }

   // load the tracks/traj association maps
   std::vector<TrajTrackAssociationCollection> TrajToTrackMap;
   Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
   for(std::vector<edm::EDGetTokenT<TrajTrackAssociationCollection> >::const_iterator token = trajTrackAssoTokens_.begin();token!=trajTrackAssoTokens_.end();++token) {
     try {iEvent.getByToken(*token,trajTrackAssociationHandle);} catch ( cms::Exception& ) {;}
     TrajToTrackMap.push_back(*trajTrackAssociationHandle.product());
   }

   // sanity check
   if(!(trackCollection.size()>0 && trajectoryCollection.size()>0)) return;

   // build the reverse map tracks -> vertex
   std::vector<std::map<size_t,int> > trackVertices;
   for(size_t i=0;i<trackSize;++i) {
     trackVertices.push_back(inVertex(trackCollection[0], vertexColl, globalvertexid_+1));
   }

   // iterate over vertices
   if(functionality_vertices_) {
     for(reco::VertexCollection::const_iterator v=vertexColl.begin();
         v!=vertexColl.end(); ++v) {
       nTracks_pvtx_ = v->tracksSize();
       sumptsq_pvtx_ = sumPtSquared(*v);
       isValid_pvtx_ = int(v->isValid());
       isFake_pvtx_ =  int(v->isFake());
       recx_pvtx_ = v->x();
       recy_pvtx_ = v->y();
       recz_pvtx_ = v->z();
       recx_err_pvtx_ = v->xError();
       recy_err_pvtx_ = v->yError();
       recz_err_pvtx_ = v->zError();
       globalvertexid_++;
       vertices_->Fill();
     }
   }

   // iterate over pixel vertices
   if(functionality_pixvertices_) {
     for(reco::VertexCollection::const_iterator v=pixelVertexColl.begin();
         v!=pixelVertexColl.end(); ++v) {
       nTracks_pvtx_ = v->tracksSize();
       sumptsq_pvtx_ = sumPtSquared(*v);
       isValid_pvtx_ = int(v->isValid());
       isFake_pvtx_ =  int(v->isFake());
       recx_pvtx_ = v->x();
       recy_pvtx_ = v->y();
       recz_pvtx_ = v->z();
       recx_err_pvtx_ = v->xError();
       recy_err_pvtx_ = v->yError();
       recz_err_pvtx_ = v->zError();
       pixelVertices_->Fill();
     }
   }

   // determine if each cluster is on a track or not, and record the local angle
   // to do this, we use the first track/traj collection
   std::vector<double> clusterOntrackAngles = onTrackAngles(clusters,trajectoryCollection[0]);
   std::vector<std::pair<double,double> > pixclusterOntrackAngles = onTrackAngles(pixelclusters,trajectoryCollection[0]);

/*
  // iterate over trajectories
  // note: when iterating over trajectories, it might be simpler to use the tracks/trajectories association map
  for(std::vector<Trajectory>::const_iterator traj = trajVec.begin(); traj< trajVec.end(); ++traj) {
  }
  // loop over all rechits from trajectories
  //iterate over trajectories
  for(std::vector<Trajectory>::const_iterator traj = trajVec.begin(); traj< trajVec.end(); ++traj) {
    Trajectory::DataContainer measurements = traj->measurements();
    // iterate over measurements
    for(Trajectory::DataContainer::iterator meas = measurements.begin(); meas!= measurements.end(); ++meas) {
    }
  }
*/

   // determine if each cluster is on a track or not, and record the trackid
   std::vector< std::vector<int> > stripClusterOntrackIndices;
   for(size_t i = 0; i<trackSize; ++i) {
     stripClusterOntrackIndices.push_back(onTrack(clusters,trackCollection[i],globaltrackid_[i]+1));
   }
   std::vector< std::vector<int> > pixelClusterOntrackIndices;
   for(size_t i = 0; i<trackSize; ++i) {
     pixelClusterOntrackIndices.push_back(onTrack(pixelclusters,trackCollection[i],globaltrackid_[i]+1));
   }
   nclustersOntrack_    = count_if(stripClusterOntrackIndices[0].begin(),stripClusterOntrackIndices[0].end(),bind2nd(not_equal_to<int>(), -1));
   npixClustersOntrack_ = count_if(pixelClusterOntrackIndices[0].begin(),pixelClusterOntrackIndices[0].end(),bind2nd(not_equal_to<int>(), -1));

   // iterate over tracks
   for (size_t coll = 0; coll<trackCollection.size(); ++coll) {
     uint32_t n_hits_barrel=0;
     uint32_t n_hits_lowprob=0;
     for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap[coll].begin(); it!=TrajToTrackMap[coll].end(); ++it) {
       reco::TrackRef itTrack  = it->val;
       edm::Ref<std::vector<Trajectory> > traj  = it->key; // bug to find type of the key
       eta_   = itTrack->eta();
       phi_   = itTrack->phi();
       try { // not all track collections have the dedx info... indeed at best one.
         dedxNoM_ = dEdxTrack1[itTrack].numberOfMeasurements();
         dedx1_ = dEdxTrack1[itTrack].dEdx();
         dedx2_ = dEdxTrack2[itTrack].dEdx();
         dedx3_ = dEdxTrack3[itTrack].dEdx();
       } catch ( cms::Exception& ) {
         dedxNoM_ = 0;
	 dedx1_ = 0.;
	 dedx2_ = 0.;
	 dedx3_ = 0.;
       }
       charge_ = itTrack->charge();
       quality_ = itTrack->qualityMask();
       foundhits_ = itTrack->found();
       lostHits_  = itTrack->lost();
       foundhitsStrips_ =  itTrack->hitPattern().numberOfValidStripHits();
       foundhitsPixels_ =  itTrack->hitPattern().numberOfValidPixelHits();
       losthitsStrips_  =  itTrack->hitPattern().numberOfLostStripHits(reco::HitPattern::TRACK_HITS);
       losthitsPixels_  =  itTrack->hitPattern().numberOfLostPixelHits(reco::HitPattern::TRACK_HITS);
       nLayers_ = uint32_t(itTrack->hitPattern().trackerLayersWithMeasurement());
       p_ = itTrack->p();
       pt_ = itTrack->pt();
       chi2_  = itTrack->chi2();
       ndof_ = (uint32_t)itTrack->ndof();
       dz_ = itTrack->dz();
       dzerr_ = itTrack->dzError();
       dzCorr_ = itTrack->dz(beamSpot);
       dxy_ = itTrack->dxy();
       dxyerr_ = itTrack->dxyError();
       dxyCorr_ = itTrack->dxy(beamSpot);
       pterr_ = itTrack->ptError();
       etaerr_ = itTrack->etaError();
       phierr_ = itTrack->phiError();
       qoverp_ = itTrack->qoverp();
       xPCA_ = itTrack->vertex().x();
       yPCA_ = itTrack->vertex().y();
       zPCA_ = itTrack->vertex().z();
       try { // only one track collection (at best) is connected to the main vertex
         if(vertexColl.size()>0 && !vertexColl.begin()->isFake()) {
           trkWeightpvtx_ =  vertexColl.begin()->trackWeight(itTrack);
         } else
	   trkWeightpvtx_ = 0.;
       } catch ( cms::Exception& ) {
         trkWeightpvtx_ = 0.;
       }
       globaltrackid_[coll]++;
       std::map<size_t,int>::const_iterator theV = trackVertices[coll].find(itTrack.key());
       vertexid_ = (theV!=trackVertices[coll].end()) ? theV->second : 0;
       // add missing hits (separate tree, common strip + pixel)
       Trajectory::DataContainer const & measurements = traj->measurements();
       if(functionality_missingHits_) {
         for(Trajectory::DataContainer::const_iterator it = measurements.begin(); it!=measurements.end(); ++it) {
           TrajectoryMeasurement::ConstRecHitPointer rechit = it->recHit();
           if(!rechit->isValid()) {
             // detid
             detid_ = rechit->geographicalId();
             // status
             type_ = rechit->getType();
             // position
             LocalPoint local = it->predictedState().localPosition();
             clPositionX_ = local.x();
             clPositionY_ = local.y();
             // global position
             GlobalPoint global = it->predictedState().globalPosition();
             globalX_ = global.x();
             globalY_ = global.y();
             globalZ_ = global.z();
             // position in the measurement frame
             measX_ = 0;
             measY_ = 0;
             if(type_ != TrackingRecHit::inactive ) {
               const GeomDetUnit* gdu = static_cast<const GeomDetUnit*>(tracker_->idToDetUnit(detid_));
               if(gdu && gdu->type().isTracker()) {
                 const Topology& topo = gdu->topology();
                 MeasurementPoint meas = topo.measurementPosition(local);
                 measX_ = meas.x();
                 measY_ = meas.y();
               }
             }
             // local error
             LocalError error = it->predictedState().localError().positionError();
             errorX_ = error.xx();
             errorY_ = error.yy();
             // fill
             missingHits_[coll]->Fill();
           }
         }
       }
       // compute the fraction of low probability pixels... will be added to the event tree
       for(trackingRecHit_iterator it = itTrack->recHitsBegin(); it!=itTrack->recHitsEnd(); ++it) {
            const TrackingRecHit* hit = &(**it);
	    const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit);
	    if(pixhit) {
	       DetId detId = pixhit->geographicalId();
	       if(detId.subdetId()==PixelSubdetector::PixelBarrel) {
	         ++n_hits_barrel;
		 double proba = pixhit->clusterProbability(0);
		 if(proba<=0.0) ++n_hits_lowprob;
	       }
	    }
       }
       // fill the track tree
       if(functionality_tracks_) tracks_[coll]->Fill();
     }
     lowPixelProbabilityFraction_[coll] = n_hits_barrel>0 ? (float)n_hits_lowprob/n_hits_barrel : -1.;
   }

   // iterate over clusters
   nclusters_ = 0;
   std::vector<double>::const_iterator angleIt = clusterOntrackAngles.begin();
   uint32_t localCounter = 0;
   for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=clusters->begin(); DSViter!=clusters->end();DSViter++ ) {
     edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
     edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
     uint32_t detid = DSViter->id();
     nclusters_ += DSViter->size();
     if(functionality_offtrackClusters_||functionality_ontrackClusters_) {
       for(edmNew::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter,++angleIt,++localCounter) {
         SiStripClusterInfo* siStripClusterInfo = new SiStripClusterInfo(*iter,iSetup,detid,std::string("")); //string = quality label
         // general quantities
         for(size_t i=0; i< trackSize; ++i) {
           trackid_[i] = stripClusterOntrackIndices[i][localCounter];
         }
         onTrack_ = (trackid_[0] != (uint32_t)-1);
         clWidth_ = siStripClusterInfo->width();
         clPosition_ = siStripClusterInfo->baryStrip();
         angle_ = *angleIt;
         thickness_ =  ((((DSViter->id()>>25)&0x7f)==0xd) ||
                       ((((DSViter->id()>>25)&0x7f)==0xe) && (((DSViter->id()>>5)&0x7)>4))) ? 500 : 300;
         stripLength_ = static_cast<const StripGeomDetUnit*>(tracker_->idToDet(detid))->specificTopology().stripLength();
         int nstrips  = static_cast<const StripGeomDetUnit*>(tracker_->idToDet(detid))->specificTopology().nstrips();
         maxCharge_ = siStripClusterInfo->maxCharge();
         // signal and noise with gain corrections
         clNormalizedCharge_ = siStripClusterInfo->charge() ;
         clNormalizedNoise_  = siStripClusterInfo->noiseRescaledByGain() ;
         clSignalOverNoise_  = siStripClusterInfo->signalOverNoise() ;
         // signal and noise with gain corrections and angle corrections
         clCorrectedCharge_ = clNormalizedCharge_ * fabs(cos(angle_)); // corrected for track angle
         clCorrectedSignalOverNoise_ = clSignalOverNoise_ * fabs(cos(angle_)); // corrected for track angle
         // signal and noise without gain corrections
         clBareNoise_  = siStripClusterInfo->noise();
         clBareCharge_ = clSignalOverNoise_*clBareNoise_;
         // global position
         const StripGeomDetUnit* sgdu = static_cast<const StripGeomDetUnit*>(tracker_->idToDet(detid));
         Surface::GlobalPoint gp = sgdu->surface().toGlobal(sgdu->specificTopology().localPosition(MeasurementPoint(clPosition_,0)));
         globalX_ = gp.x();
         globalY_ = gp.y();
         globalZ_ = gp.z();
         // cabling
         detid_ = detid;
         lldChannel_ = 1+(int(floor(iter->barycenter()))/256);
         if(lldChannel_==2 && nstrips==512) lldChannel_=3;
         if((functionality_offtrackClusters_&&!onTrack_)||(functionality_ontrackClusters_&&onTrack_)) clusters_->Fill();
         delete siStripClusterInfo;
       }
     }
   }

   // iterate over pixel clusters
   npixClusters_ = 0;
   std::vector<std::pair<double,double> >::const_iterator pixAngleIt = pixclusterOntrackAngles.begin();
   localCounter = 0;
   for (edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=pixelclusters->begin(); DSViter!=pixelclusters->end();DSViter++ ) {
     edmNew::DetSet<SiPixelCluster>::const_iterator begin=DSViter->begin();
     edmNew::DetSet<SiPixelCluster>::const_iterator end  =DSViter->end();
     uint32_t detid = DSViter->id();
     npixClusters_ += DSViter->size();
     if(functionality_pixclusters_) {
       for(edmNew::DetSet<SiPixelCluster>::const_iterator iter=begin;iter!=end;++iter,++pixAngleIt,++localCounter) {
         // general quantities
         for(size_t i=0; i< trackSize; ++i) {
           trackid_[i] = pixelClusterOntrackIndices[i][localCounter];
         }
         onTrack_ = (trackid_[0] != (uint32_t)-1);
         clPositionX_ = iter->x();
         clPositionY_ = iter->y();
         clSize_  = iter->size();
         clSizeX_ = iter->sizeX();
         clSizeY_ = iter->sizeY();
         alpha_ = pixAngleIt->first;
         beta_  = pixAngleIt->second;
         charge_ = (iter->charge())/1000.;
         chargeCorr_ = charge_ * sqrt( 1.0 / ( 1.0/pow( tan(alpha_), 2 ) + 1.0/pow( tan(beta_), 2 ) + 1.0 ))/1000.;
         // global position
         const PixelGeomDetUnit* pgdu = static_cast<const PixelGeomDetUnit*>(tracker_->idToDet(detid));
         Surface::GlobalPoint gp = pgdu->surface().toGlobal(pgdu->specificTopology().localPosition(MeasurementPoint(clPositionX_,clPositionY_)));
         globalX_ = gp.x();
         globalY_ = gp.y();
         globalZ_ = gp.z();
         // cabling
         detid_ = detid;
         // fill
         pixclusters_->Fill();
       }
     }
   }

   // topological quantities - uses the first track collection
   EventShape shape(trackCollection[0]);
   math::XYZTLorentzVectorF thrust = shape.thrust();
   thrustValue_ = thrust.t();
   thrustX_ = thrust.x();
   thrustY_ = thrust.y();
   thrustZ_ = thrust.z();
   sphericity_ = shape.sphericity();
   planarity_  = shape.planarity();
   aplanarity_ = shape.aplanarity();

   // fill event tree
   if(functionality_events_) event_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void
TrackerDpgAnalysis::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
   const TrackerTopology* const tTopo = tTopoHandle.product();

   //geometry
   iSetup.get<TrackerDigiGeometryRecord>().get(tracker_);

   //HLT names
   bool changed (true);
   if (hltConfig_.init(iRun,iSetup,HLTTag_.process(),changed)) {
     if (changed) {
       hlNames_=hltConfig_.triggerNames();
     }
   }
   int i=0;
   for(std::vector<std::string>::const_iterator it = hlNames_.begin(); it<hlNames_.end();++it) {
     std::cout << (i++) << " = " << (*it) << std::endl;
   }

   // read the delay offsets for each device from input files
   // this is only for the so-called "random delay" run
   std::map<uint32_t,float> delayMap = delay(delayFileNames_);
   TrackerMap tmap("Delays");

   // cabling I (readout)
   iSetup.get<SiStripFedCablingRcd>().get( cabling_ );
   auto feds = cabling_->fedIds() ;
   for(auto fedid = feds.begin();fedid<feds.end();++fedid) {
     auto connections = cabling_->fedConnections(*fedid);
     for(auto conn=connections.begin();conn<connections.end();++conn) {
       // Fill the "old" map to be used for lookup during analysis
       if(conn->isConnected())
         connections_.insert(std::make_pair(conn->detId(),new FedChannelConnection(*conn)));
       // Fill the standalone tree (once for all)
       if(conn->isConnected()) {
         detid_ = conn->detId();
         strncpy(moduleName_,toStringName(detid_,tTopo).c_str(),256);
         strncpy(moduleId_,toStringId(detid_).c_str(),256);
         lldChannel_ = conn->lldChannel();
         dcuId_ = conn->dcuId();
         fecCrate_ = conn->fecCrate();
         fecSlot_ = conn->fecSlot();
         fecRing_ = conn->fecRing();
         ccuAdd_ = conn->ccuAddr();
         ccuChan_ = conn->ccuChan();
         fedId_ = conn->fedId();
         fedCh_ = conn->fedCh();
         fiberLength_ = conn->fiberLength();
	 delay_ = delayMap[dcuId_];
	 const StripGeomDetUnit* sgdu = static_cast<const StripGeomDetUnit*>(tracker_->idToDet(detid_));
	 Surface::GlobalPoint gp = sgdu->surface().toGlobal(LocalPoint(0,0));
	 globalX_ = gp.x();
	 globalY_ = gp.y();
	 globalZ_ = gp.z();
	 readoutmap_->Fill();
         tmap.fill_current_val(detid_,delay_);
       }
     }
   }
   if(delayMap.size()) tmap.save(true, 0, 0, "delaymap.png");

   // cabling II (DCU map)
   std::ifstream cablingFile(cablingFileName_.c_str());
   if(cablingFile.is_open()) {
     char buffer[1024];
     cablingFile.getline(buffer,1024);
     while(!cablingFile.eof()) {
       std::istringstream line(buffer);
       std::string name;
       // one line contains the PSU name + all dcuids connected to it.
       line >> name;
       strncpy(PSUname_,name.c_str(),256);
       while(!line.eof()) {
         line >> dcuId_;
         psumap_->Fill();
       }
       cablingFile.getline(buffer,1024);
     }
   } else {
     edm::LogWarning("BadConfig") << " The PSU file does not exist. The psumap tree will not be filled."
                                  << std::endl << " Looking for " << cablingFileName_.c_str() << "."
				  << std::endl << " Please specify a valid filename through the PSUFileName untracked parameter.";
   }
}

// ------------ method called once each job just after ending the event loop  ------------
void
TrackerDpgAnalysis::endJob() {
  for(size_t i = 0; i<tracks_.size();++i) {
    char buffer[256];
    sprintf(buffer,"trackid%lu",(unsigned long)i);
    if(tracks_[i]->GetEntries()) tracks_[i]->BuildIndex(buffer,"eventid");
  }
  /* not needed: missing hits is a high-level quantity
  for(size_t i = 0; i<missingHits_.size();++i) {
    char buffer[256];
    sprintf(buffer,"trackid%lu",(unsigned long)i);
    if(missingHits_[i]->GetEntries()) missingHits_[i]->BuildIndex(buffer);
  }
  */
  if(vertices_->GetEntries()) vertices_->BuildIndex("vertexid","eventid");
  if(event_->GetEntries()) event_->BuildIndex("runid","eventid");
  if(psumap_->GetEntries()) psumap_->BuildIndex("dcuId");
  if(readoutmap_->GetEntries()) readoutmap_->BuildIndex("detid","lldChannel");
}

std::vector<double> TrackerDpgAnalysis::onTrackAngles(edm::Handle<edmNew::DetSetVector<SiStripCluster> >& clusters,
                                                       const std::vector<Trajectory>& trajVec )
{
  std::vector<double> result;
  // first, build a list of positions and angles on trajectories
  std::multimap<const uint32_t,std::pair<LocalPoint,double> > onTrackPositions;
  for(std::vector<Trajectory>::const_iterator traj = trajVec.begin(); traj< trajVec.end(); ++traj) {
    Trajectory::DataContainer measurements = traj->measurements();
    for(Trajectory::DataContainer::iterator meas = measurements.begin(); meas!= measurements.end(); ++meas) {
      double tla = meas->updatedState().localDirection().theta();
      insertMeasurement(onTrackPositions,&(*(meas->recHit())),tla);
    }
  }
  // then loop over the clusters to check
  double angle = 0.;
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=clusters->begin(); DSViter!=clusters->end();DSViter++ ) {
    edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
    std::pair< std::multimap<uint32_t,std::pair<LocalPoint,double> >::const_iterator,
               std::multimap<uint32_t,std::pair<LocalPoint,double> >::const_iterator> range =
      onTrackPositions.equal_range(DSViter->id());
    const GeomDetUnit* gdu = static_cast<const GeomDetUnit*>(tracker_->idToDet(DSViter->id()));
    for(edmNew::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter) {
      angle = 0.;
      for(std::multimap<uint32_t,std::pair<LocalPoint,double> >::const_iterator cl = range.first; cl!= range.second; ++cl) {
        if(fabs(gdu->topology().measurementPosition(cl->second.first).x()-iter->barycenter())<2) {
	  angle = cl->second.second;
	}
      }
      result.push_back(angle);
    }
  }
  return result;
}

void TrackerDpgAnalysis::insertMeasurement(std::multimap<const uint32_t,std::pair<LocalPoint,double> >& collection,const TransientTrackingRecHit* hit , double tla)
{
      if(!hit) return;
      const SiTrackerMultiRecHit* multihit=dynamic_cast<const SiTrackerMultiRecHit*>(hit);
      const SiStripRecHit2D* singlehit=dynamic_cast<const SiStripRecHit2D*>(hit);
      const SiStripRecHit1D* hit1d=dynamic_cast<const SiStripRecHit1D*>(hit);
      if(hit1d) { //...->33X
        collection.insert(std::make_pair(hit1d->geographicalId().rawId(),std::make_pair(hit1d->localPosition(),tla)));
      } else if(singlehit) { // 41X->...
        collection.insert(std::make_pair(singlehit->geographicalId().rawId(),std::make_pair(singlehit->localPosition(),tla)));
      }
      else if(multihit){
        std::vector< const TrackingRecHit * > childs = multihit->recHits();
	for(std::vector<const TrackingRecHit*>::const_iterator it=childs.begin();it!=childs.end();++it) {
	   insertMeasurement(collection,dynamic_cast<const TrackingRecHit*>(*it),tla);
	}
      }
}

std::vector<int> TrackerDpgAnalysis::onTrack(edm::Handle<edmNew::DetSetVector<SiStripCluster> >& clusters,
                                              const reco::TrackCollection& trackVec, uint32_t firstTrack )
{
  std::vector<int> result;
  // first, build a list of positions and trackid on tracks
  std::multimap<const uint32_t,std::pair<int,int> > onTrackPositions;
  uint32_t trackid = firstTrack;
  for(reco::TrackCollection::const_iterator itTrack = trackVec.begin(); itTrack!=trackVec.end();++itTrack,++trackid) {
    for(trackingRecHit_iterator it = itTrack->recHitsBegin(); it!=itTrack->recHitsEnd(); ++it) {
      const TrackingRecHit* hit = &(**it);
      insertMeasurement(onTrackPositions,hit,trackid);
    }
  }
  // then loop over the clusters to check
  int thetrackid = -1;
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=clusters->begin(); DSViter!=clusters->end();DSViter++ ) {
    edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
    std::pair< std::multimap<uint32_t,std::pair<int,int> >::const_iterator,
               std::multimap<uint32_t,std::pair<int,int> >::const_iterator> range =
      onTrackPositions.equal_range(DSViter->id());
    for(edmNew::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter) {
      thetrackid = -1;
      for(std::multimap<uint32_t,std::pair<int,int> >::const_iterator cl = range.first; cl!= range.second; ++cl) {
        if(fabs(cl->second.first-iter->barycenter())<2) {
	  thetrackid = cl->second.second;
	}
      }
      result.push_back(thetrackid);
    }
  }
  return result;
}

void TrackerDpgAnalysis::insertMeasurement(std::multimap<const uint32_t,std::pair<int, int> >& collection,const TrackingRecHit* hit , int trackid)
{
      if(!hit) return;
      const SiTrackerMultiRecHit* multihit=dynamic_cast<const SiTrackerMultiRecHit*>(hit);
      const SiStripRecHit2D* singlehit=dynamic_cast<const SiStripRecHit2D*>(hit);
      const SiStripRecHit1D* hit1d=dynamic_cast<const SiStripRecHit1D*>(hit);
      if(hit1d) { // 41X->...
        collection.insert(std::make_pair(hit1d->geographicalId().rawId(),std::make_pair(int(hit1d->cluster()->barycenter()),trackid)));
      } else if(singlehit) { //...->33X
        collection.insert(std::make_pair(singlehit->geographicalId().rawId(),std::make_pair(int(singlehit->cluster()->barycenter()),trackid)));
      }
      else if(multihit){
        std::vector< const TrackingRecHit * > childs = multihit->recHits();
	for(std::vector<const TrackingRecHit*>::const_iterator it=childs.begin();it!=childs.end();++it) {
	   insertMeasurement(collection,*it,trackid);
	}
      }
}

std::map<size_t,int> TrackerDpgAnalysis::inVertex(const reco::TrackCollection& tracks, const reco::VertexCollection& vertices, uint32_t firstVertex)
{
  // build reverse map track -> vertex
  std::map<size_t,int> output;
  uint32_t vertexid = firstVertex;
  for(reco::VertexCollection::const_iterator v = vertices.begin(); v!=vertices.end(); ++v,++vertexid) {
    reco::Vertex::trackRef_iterator it = v->tracks_begin();
    reco::Vertex::trackRef_iterator lastTrack = v->tracks_end();
    for(;it!=lastTrack;++it) {
      output[it->key()] = vertexid;
    }
  }
  return output;
}

std::vector<std::pair<double,double> > TrackerDpgAnalysis::onTrackAngles(edm::Handle<edmNew::DetSetVector<SiPixelCluster> >& clusters,
                                                                          const std::vector<Trajectory>& trajVec )
{
  std::vector<std::pair<double,double> > result;
  // first, build a list of positions and angles on trajectories
  std::multimap<const uint32_t,std::pair<LocalPoint,std::pair<double,double> > > onTrackPositions;
  for(std::vector<Trajectory>::const_iterator traj = trajVec.begin(); traj< trajVec.end(); ++traj) {
    Trajectory::DataContainer measurements = traj->measurements();
    for(Trajectory::DataContainer::iterator meas = measurements.begin(); meas!= measurements.end(); ++meas) {
      LocalVector localDir = meas->updatedState().localDirection();
      double alpha = atan2(localDir.z(), localDir.x());
      double beta  = atan2(localDir.z(), localDir.y());
      insertMeasurement(onTrackPositions,&(*(meas->recHit())),alpha,beta);
    }
  }
  // then loop over the clusters to check
  double alpha = 0.;
  double beta  = 0.;
  for (edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=clusters->begin(); DSViter!=clusters->end();DSViter++ ) {
    edmNew::DetSet<SiPixelCluster>::const_iterator begin=DSViter->begin();
    edmNew::DetSet<SiPixelCluster>::const_iterator end  =DSViter->end();
    for(edmNew::DetSet<SiPixelCluster>::const_iterator iter=begin;iter!=end;++iter) {
      alpha = 0.;
      beta  = 0.;
      std::pair< std::multimap<uint32_t,std::pair<LocalPoint,std::pair<double, double> > >::const_iterator,
                 std::multimap<uint32_t,std::pair<LocalPoint,std::pair<double, double> > >::const_iterator> range =
          onTrackPositions.equal_range(DSViter->id());
      const GeomDetUnit* gdu = static_cast<const GeomDetUnit*>(tracker_->idToDet(DSViter->id()));
      for(std::multimap<uint32_t,std::pair<LocalPoint,std::pair<double, double> > >::const_iterator cl = range.first; cl!= range.second; ++cl) {
        if(fabs(gdu->topology().measurementPosition(cl->second.first).x()-iter->x())<2 &&
	   fabs(gdu->topology().measurementPosition(cl->second.first).y()-iter->y())<2    ) {
	  alpha = cl->second.second.first;
	  beta  = cl->second.second.second;
	}
      }
      result.push_back(std::make_pair(alpha,beta));
    }
  }
  return result;
}

void TrackerDpgAnalysis::insertMeasurement(std::multimap<const uint32_t,std::pair<LocalPoint,std::pair<double,double> > >& collection,const TransientTrackingRecHit* hit , double alpha, double beta)
{
      if(!hit) return;
      const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit);
      if(pixhit) {
        collection.insert(std::make_pair(pixhit->geographicalId().rawId(),std::make_pair(pixhit->localPosition(),std::make_pair(alpha,beta))));
      }
}

std::vector<int> TrackerDpgAnalysis::onTrack(edm::Handle<edmNew::DetSetVector<SiPixelCluster> >& clusters,
                                              const reco::TrackCollection& trackVec, uint32_t firstTrack )
{
  std::vector<int> result;
  // first, build a list of positions and trackid on tracks
  std::multimap<const uint32_t,std::pair<std::pair<float, float>,int> > onTrackPositions;
  uint32_t trackid = firstTrack;
  for(reco::TrackCollection::const_iterator itTrack = trackVec.begin(); itTrack!=trackVec.end();++itTrack,++trackid) {
    for(trackingRecHit_iterator it = itTrack->recHitsBegin(); it!=itTrack->recHitsEnd(); ++it) {
      const TrackingRecHit* hit = &(**it);
      insertMeasurement(onTrackPositions,hit,trackid);
    }
  }
  // then loop over the clusters to check
  int thetrackid = -1;
  for (edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=clusters->begin(); DSViter!=clusters->end();DSViter++ ) {
    edmNew::DetSet<SiPixelCluster>::const_iterator begin=DSViter->begin();
    edmNew::DetSet<SiPixelCluster>::const_iterator end  =DSViter->end();
    for(edmNew::DetSet<SiPixelCluster>::const_iterator iter=begin;iter!=end;++iter) {
      thetrackid = -1;
      std::pair< std::multimap<uint32_t,std::pair<std::pair<float, float>,int> >::const_iterator,
                 std::multimap<uint32_t,std::pair<std::pair<float, float>,int> >::const_iterator> range =
          onTrackPositions.equal_range(DSViter->id());
      for(std::multimap<uint32_t,std::pair<std::pair<float, float>,int> >::const_iterator cl = range.first; cl!= range.second; ++cl) {
        if((fabs(cl->second.first.first-iter->x())<2)&&(fabs(cl->second.first.second-iter->y())<2)) {
	  thetrackid = cl->second.second;
	}
      }
      result.push_back(thetrackid);
    }
  }
  return result;
}

void TrackerDpgAnalysis::insertMeasurement(std::multimap<const uint32_t,std::pair<std::pair<float, float>, int> >& collection,const TrackingRecHit* hit , int trackid)
{
      if(!hit) return;
      const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit);
      if(pixhit) {
        collection.insert(std::make_pair(pixhit->geographicalId().rawId(),std::make_pair(std::make_pair(pixhit->cluster()->x(),pixhit->cluster()->y()),trackid)));
	}
}

std::string TrackerDpgAnalysis::toStringName(uint32_t rawid, const TrackerTopology* tTopo) {
   SiStripDetId detid(rawid);
   std::string out;
   std::stringstream output;
   switch(detid.subDetector()) {
     case 3:
     {
       output << "TIB";

       output << (tTopo->tibIsZPlusSide(rawid) ? "+" : "-");
       output << " layer ";
       output << tTopo->tibLayer(rawid);
       output << ", string ";
       output << tTopo->tibString(rawid);
       output << (tTopo->tibIsExternalString(rawid) ? " external" : " internal");
       output << ", module ";
       output << tTopo->tibModule(rawid);
       if(tTopo->tibIsDoubleSide(rawid)) {
         output << " (double)";
       } else {
         output << (tTopo->tibIsRPhi(rawid) ? " (rphi)" : " (stereo)");
       }
       break;
     }
     case 4:
     {
       output << "TID";

       output << (tTopo->tidIsZPlusSide(rawid) ? "+" : "-");
       output << " disk ";
       output << tTopo->tidWheel(rawid);
       output << ", ring ";
       output << tTopo->tidRing(rawid);
       output << (tTopo->tidIsFrontRing(rawid) ? " front" : " back");
       output << ", module ";
       output << tTopo->tidModule(rawid);
       if(tTopo->tidIsDoubleSide(rawid)) {
         output << " (double)";
       } else {
         output << (tTopo->tidIsRPhi(rawid) ? " (rphi)" : " (stereo)");
       }
       break;
     }
     case 5:
     {
       output << "TOB";

       output << (tTopo->tobIsZPlusSide(rawid) ? "+" : "-");
       output << " layer ";
       output << tTopo->tobLayer(rawid);
       output << ", rod ";
       output << tTopo->tobRod(rawid);
       output << ", module ";
       output << tTopo->tobModule(rawid);
       if(tTopo->tobIsDoubleSide(rawid)) {
         output << " (double)";
       } else {
         output << (tTopo->tobIsRPhi(rawid) ? " (rphi)" : " (stereo)");
       }
       break;
     }
     case 6:
     {
       output << "TEC";

       output << (tTopo->tecIsZPlusSide(rawid) ? "+" : "-");
       output << " disk ";
       output << tTopo->tecWheel(rawid);
       output << " sector ";
       output << tTopo->tecPetalNumber(rawid);
       output << (tTopo->tecIsFrontPetal(rawid) ? " Front Petal" : " Back Petal");
       output << ", module ";
       output << tTopo->tecRing(rawid);
       output << tTopo->tecModule(rawid);
       if(tTopo->tecIsDoubleSide(rawid)) {
         output << " (double)";
       } else {
         output << (tTopo->tecIsRPhi(rawid) ? " (rphi)" : " (stereo)");
       }
       break;
     }
     default:
     {
       output << "UNKNOWN";
     }
   }
   out = output.str();
   return out;
}

std::string TrackerDpgAnalysis::toStringId(uint32_t rawid) {
   std::string out;
   std::stringstream output;
   output << rawid << " (0x" << std::hex << rawid << std::dec << ")";
   out = output.str();
   return out;
}

double TrackerDpgAnalysis::sumPtSquared(const reco::Vertex & v)  {
  double sum = 0.;
  double pT;
  for (reco::Vertex::trackRef_iterator it = v.tracks_begin(); it != v.tracks_end(); it++) {
    pT = (**it).pt();
    sum += pT*pT;
  }
  return sum;
}

float TrackerDpgAnalysis::delay(const SiStripEventSummary& summary) {
  float delay = const_cast<SiStripEventSummary&>(summary).ttcrx();
  uint32_t latencyCode = (const_cast<SiStripEventSummary&>(summary).layerScanned()>>24)&0xff;
  int latencyShift = latencyCode & 0x3f;             // number of bunch crossings between current value and start of scan... must be positive
  if(latencyShift>32) latencyShift -=64;             // allow negative values: we cover [-32,32].. should not be needed.
  if((latencyCode>>6)==2) latencyShift -= 3;         // layer in deconv, rest in peak
  if((latencyCode>>6)==1) latencyShift += 3;         // layer in peak, rest in deconv
  float correctedDelay = delay - (latencyShift*25.); // shifts the delay so that 0 corresponds to the current settings.
  return correctedDelay;
}

std::map<uint32_t,float> TrackerDpgAnalysis::delay(const std::vector<std::string>& files) {
   // prepare output
   uint32_t dcuid;
   float delay;
   std::map<uint32_t,float> delayMap;
   //iterator over input files
   for(std::vector<std::string>::const_iterator file=files.begin();file<files.end();++file){
     // open the file
     std::ifstream cablingFile(file->c_str());
     if(cablingFile.is_open()) {
       char buffer[1024];
       // read one line
       cablingFile.getline(buffer,1024);
       while(!cablingFile.eof()) {
         std::string line(buffer);
         size_t pos = line.find("dcuid");
         // one line containing dcuid
         if(pos != std::string::npos) {
           // decode dcuid
           std::string dcuids = line.substr(pos+7,line.find(" ",pos)-pos-8);
           std::istringstream dcuidstr(dcuids);
           dcuidstr >> std::hex >> dcuid;
           // decode delay
           pos = line.find("difpll");
           std::string diffs = line.substr(pos+8,line.find(" ",pos)-pos-9);
           std::istringstream diffstr(diffs);
           diffstr >> delay;
           // fill the map
           delayMap[dcuid] = delay;
         }
         // iterate
         cablingFile.getline(buffer,1024);
       }
     } else {
       edm::LogWarning("BadConfig") << " The delay file does not exist. The delay map will not be filled properly."
                                    << std::endl << " Looking for " << file->c_str() << "."
          			    << std::endl << " Please specify valid filenames through the DelayFileNames untracked parameter.";
     }
   }
   return delayMap;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerDpgAnalysis);
