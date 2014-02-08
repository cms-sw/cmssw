#ifndef TrackingMonitor_H
#define TrackingMonitor_H
// -*- C++ -*-
//
// Package:    TrackingMonitor
// Class:      TrackingMonitor
// 
/**\class TrackingMonitor TrackingMonitor.cc DQM/TrackerMonitorTrack/src/TrackingMonitor.cc
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Suchandra Dutta, Giorgia Mila
//         Created:  Thu 28 22:45:30 CEST 2008

#include <memory>
#include <fstream>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h" 

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class DQMStore;
class TrackAnalyzer;
class TrackBuildingAnalyzer;
class VertexMonitor;
class GetLumi;
class TProfile;
class GenericTriggerEventFlag;

class TrackingMonitor : public edm::EDAnalyzer 
{
    public:
        explicit TrackingMonitor(const edm::ParameterSet&);
        ~TrackingMonitor();
        virtual void beginJob(void);
        virtual void endJob(void);

	virtual void setMaxMinBin(std::vector<double> & ,std::vector<double> &  ,std::vector<int> &  ,double, double, int, double, double, int);
	virtual void setNclus(const edm::Event&, std::vector<int> & );

        virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup&  eSetup);
        virtual void analyze(const edm::Event&, const edm::EventSetup&);
        virtual void beginRun(const edm::Run&, const edm::EventSetup&); 
        virtual void endRun(const edm::Run&, const edm::EventSetup&);

    private:
        void doProfileX(TH2 * th2, MonitorElement* me);
        void doProfileX(MonitorElement * th2m, MonitorElement* me);


        // ----------member data ---------------------------

        std::string histname;  //for naming the histograms according to algorithm used

        DQMStore * dqmStore_;

        edm::ParameterSet conf_;

        // the track analyzer
        edm::InputTag bsSrc_;
	edm::InputTag pvSrc_;
	edm::EDGetTokenT<reco::BeamSpot> bsSrcToken_;
	edm::EDGetTokenT<reco::VertexCollection> pvSrcToken_;

	edm::EDGetTokenT<reco::TrackCollection> allTrackToken_;
	edm::EDGetTokenT<reco::TrackCollection> trackToken_;
	edm::EDGetTokenT<TrackCandidateCollection> trackCandidateToken_;
	edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken_;

	edm::InputTag stripClusterInputTag_;
	edm::InputTag pixelClusterInputTag_;
	edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClustersToken_;
	edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersToken_;

	std::string Quality_;
	std::string AlgoName_;


        TrackAnalyzer * theTrackAnalyzer;
        TrackBuildingAnalyzer  * theTrackBuildingAnalyzer;
	std::vector<VertexMonitor*> theVertexMonitor;
	GetLumi*                    theLumiDetails_;

        // Tracks 
        MonitorElement * NumberOfTracks;
        MonitorElement * NumberOfMeanRecHitsPerTrack;
        MonitorElement * NumberOfMeanLayersPerTrack;  

	// Good Tracks 
        MonitorElement * FractionOfGoodTracks;

        // Track Seeds 
        MonitorElement * NumberOfSeeds;
        MonitorElement * NumberOfSeeds_lumiFlag;
	std::vector<MonitorElement *> SeedsVsClusters;
	std::vector<std::string> ClusterLabels;
	

        // Track Candidates
        MonitorElement * NumberOfTrackCandidates;

        // Cluster Properties
	std::vector<MonitorElement*> NumberOfTrkVsClusters;
        MonitorElement* NumberOfTrkVsClus;
        MonitorElement* NumberOfTrkVsStripClus;
        MonitorElement* NumberOfTrkVsPixelClus;

	// Monitoring vs LS
	MonitorElement* NumberOfTracksVsLS;
	MonitorElement* GoodTracksFractionVsLS;
	MonitorElement* NumberOfRecHitsPerTrackVsLS;

	// Monitoring PU
	MonitorElement* NumberOfTracksVsGoodPVtx;
	MonitorElement* NumberOfTracksVsBXlumi;

	// add in order to deal with LS transitions
        MonitorElement * NumberOfTracks_lumiFlag;

        std::string builderName;
        edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;
 
        bool doTrackerSpecific_; 
        bool doLumiAnalysis;
	bool doProfilesVsLS_;
	bool doAllSeedPlots;
	bool doAllPlots;
	bool doDCAPlots_;
	bool doGeneralPropertiesPlots_;
	bool doHitPropertiesPlots_;
	bool doTkCandPlots;
	bool doSeedNumberPlot;
	bool doSeedLumiAnalysis_;
	bool doSeedVsClusterPlot;
	bool runTrackBuildingAnalyzerForSeed;
	// ADD by Mia in order to have GoodTrack plots only for collision
	bool doPUmonitoring_;
	bool doPlotsVsBXlumi_;
	bool doPlotsVsGoodPVtx_;
	bool doFractionPlot_;

        GenericTriggerEventFlag* genTriggerEventFlag_;

	StringCutObjectSelector<reco::Track,true> numSelection_;
	StringCutObjectSelector<reco::Track,true> denSelection_;

};

#endif //define TrackingMonitor_H
