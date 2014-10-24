#ifndef TrackAnalyzer_H
#define TrackAnalyzer_H
// -*- C++ -*-
//
// 
/**\class TrackingAnalyzer TrackingAnalyzer.cc 
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Suchandra Dutta, Giorgia Mila
//         Created:  Thu 28 22:45:30 CEST 2008

#include <memory>
#include <fstream>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class DQMStore;

class BeamSpot;
class TrackAnalyzer 
{
    public:
        TrackAnalyzer(const edm::ParameterSet&);
	TrackAnalyzer(const edm::ParameterSet&, edm::ConsumesCollector& iC);
        virtual ~TrackAnalyzer();
        virtual void initHisto(DQMStore::IBooker & ibooker);

        virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track);

        void doSoftReset  (DQMStore * dqmStore_);
        void doReset      ();
        void undoSoftReset(DQMStore * dqmStore_);
        void setLumiFlag();

    private:
	void initHistos();
        void fillHistosForState(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname);
        void bookHistosForState(std::string sname,DQMStore::IBooker & ibooker);
        void bookHistosForHitProperties(DQMStore::IBooker & ibooker);
	void bookHistosForLScertification(DQMStore::IBooker & ibooker);
	void bookHistosForBeamSpot(DQMStore::IBooker & ibooker);
        void bookHistosForTrackerSpecific(DQMStore::IBooker & ibooker);
        void fillHistosForHitProperties(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname);
	void fillHistosForLScertification(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname);
        void fillHistosForTrackerSpecific(const reco::Track & track);

        // ----------member data ---------------------------
	std::string TopFolder_;

	edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
	edm::EDGetTokenT<reco::VertexCollection> pvToken_;
	
        edm::ParameterSet conf_;

        bool doTrackerSpecific_;
        bool doAllPlots_;
        bool doBSPlots_;
        bool doPVPlots_;
	bool doDCAPlots_;
	bool doGeneralPropertiesPlots_;
	bool doMeasurementStatePlots_;
	bool doHitPropertiesPlots_;
	bool doRecHitVsPhiVsEtaPerTrack_;
	// ADD by Mia
	bool doLayersVsPhiVsEtaPerTrack_;
	bool doTrackRecHitVsPhiVsEtaPerTrack_;
	bool doTrackLayersVsPhiVsEtaPerTrack_;
	bool doTrack2DChi2Plots_;
	bool doRecHitsPerTrackProfile_;
	// ADD by Mia in order to clean the tracking MEs
	// do not plot *Theta* and TrackPx* and TrackPy*
	bool doThetaPlots_;
	bool doTrackPxPyPlots_;
	// ADD by Mia in order to not plot DistanceOfClosestApproach w.r.t. (0,0,0)
	// the DistanceOfClosestApproach w.r.t. the beam-spot is already shown in DistanceOfClosestApproachToBS
	bool doDCAwrtPVPlots_;
	bool doDCAwrt000Plots_;

	bool doLumiAnalysis_;

	// ADD by Mia in order to turnON test MEs
	bool doTestPlots_;

        struct TkParameterMEs {
	  TkParameterMEs() :
	    TrackP(NULL)
	    , TrackPx(NULL)
	    , TrackPy(NULL)
	    , TrackPz(NULL)
	    , TrackPt(NULL)
	    
	    , TrackPxErr(NULL)
	    , TrackPyErr(NULL)
	    , TrackPzErr(NULL)
	    , TrackPtErr(NULL)
	    , TrackPErr(NULL)
	    
	    , TrackPtErrVsEta(NULL)
	    
	    , TrackQ(NULL)
	    
	    , TrackPhi(NULL)
	    , TrackEta(NULL)
	    , TrackTheta(NULL)
	    
	    , TrackPhiErr(NULL)
	    , TrackEtaErr(NULL)
	    , TrackThetaErr(NULL)
	    
	    , NumberOfRecHitsPerTrackVsPhi(NULL)
	    , NumberOfRecHitsPerTrackVsTheta(NULL)
	    , NumberOfRecHitsPerTrackVsEta(NULL)
	    , NumberOfRecHitVsPhiVsEtaPerTrack(NULL)
	    
	    , NumberOfValidRecHitsPerTrackVsPhi(NULL)
	    , NumberOfValidRecHitsPerTrackVsTheta(NULL)
	    , NumberOfValidRecHitsPerTrackVsEta(NULL)
	    , NumberOfValidRecHitVsPhiVsEtaPerTrack(NULL)

	    , NumberOfLayersPerTrackVsPhi(NULL)
	    , NumberOfLayersPerTrackVsTheta(NULL)
	    , NumberOfLayersPerTrackVsEta(NULL)

	    , Chi2oNDFVsEta(NULL)
	    , Chi2oNDFVsPhi(NULL)
	    , Chi2oNDFVsTheta(NULL)

	    , Chi2ProbVsEta(NULL)
	    , Chi2ProbVsPhi(NULL)
	    , Chi2ProbVsTheta(NULL)
	  {}
	  
	  MonitorElement* TrackP;
	  MonitorElement* TrackPx;
	  MonitorElement* TrackPy;
	  MonitorElement* TrackPz;
	  MonitorElement* TrackPt;
	  
	  MonitorElement* TrackPxErr;
	  MonitorElement* TrackPyErr;
	  MonitorElement* TrackPzErr;
	  MonitorElement* TrackPtErr;
	  MonitorElement* TrackPErr;
	  
	  MonitorElement* TrackPtErrVsEta;
	  
	  MonitorElement* TrackQ;
	  
	  MonitorElement* TrackPhi;
	  MonitorElement* TrackEta;
	  MonitorElement* TrackTheta;
	  
	  MonitorElement* TrackPhiErr;
	  MonitorElement* TrackEtaErr;
	  MonitorElement* TrackThetaErr;
	  
	  MonitorElement* NumberOfRecHitsPerTrackVsPhi;
	  MonitorElement* NumberOfRecHitsPerTrackVsTheta;
	  MonitorElement* NumberOfRecHitsPerTrackVsEta;
	  MonitorElement* NumberOfRecHitVsPhiVsEtaPerTrack;
	  
	  MonitorElement* NumberOfValidRecHitsPerTrackVsPhi;
	  MonitorElement* NumberOfValidRecHitsPerTrackVsTheta;
	  MonitorElement* NumberOfValidRecHitsPerTrackVsEta;
	  MonitorElement* NumberOfValidRecHitVsPhiVsEtaPerTrack;

	  MonitorElement* NumberOfLayersPerTrackVsPhi;
	  MonitorElement* NumberOfLayersPerTrackVsTheta;
	  MonitorElement* NumberOfLayersPerTrackVsEta;

	  MonitorElement* Chi2oNDFVsEta;
	  MonitorElement* Chi2oNDFVsPhi;
	  MonitorElement* Chi2oNDFVsTheta;
	  
	  MonitorElement* Chi2ProbVsEta;
	  MonitorElement* Chi2ProbVsPhi;
	  MonitorElement* Chi2ProbVsTheta;
	
	};
        std::map<std::string, TkParameterMEs> TkParameterMEMap;
	
	
	MonitorElement* NumberOfRecHitsPerTrack;
	MonitorElement* NumberOfValidRecHitsPerTrack;
	MonitorElement* NumberOfLostRecHitsPerTrack;
	
	MonitorElement* NumberOfRecHitsPerTrackVsPhi;
	MonitorElement* NumberOfRecHitsPerTrackVsTheta;
	MonitorElement* NumberOfRecHitsPerTrackVsEta;
	MonitorElement* NumberOfRecHitVsPhiVsEtaPerTrack;
	
	MonitorElement* NumberOfValidRecHitsPerTrackVsPhi;
	MonitorElement* NumberOfValidRecHitsPerTrackVsTheta;
	MonitorElement* NumberOfValidRecHitsPerTrackVsEta;
	MonitorElement* NumberOfValidRecHitVsPhiVsEtaPerTrack;
	
	MonitorElement* NumberOfLayersPerTrack;
	
	MonitorElement* NumberOfLayersPerTrackVsPhi;
	MonitorElement* NumberOfLayersPerTrackVsTheta;
	MonitorElement* NumberOfLayersPerTrackVsEta;

	MonitorElement* NumberOfLayersVsPhiVsEtaPerTrack;

	MonitorElement* Chi2;
	MonitorElement* Chi2Prob;
	MonitorElement* Chi2oNDF;

	MonitorElement* Chi2oNDFVsEta;
	MonitorElement* Chi2oNDFVsPhi;
	MonitorElement* Chi2oNDFVsTheta;
	
	MonitorElement* Chi2ProbVsEta;
	MonitorElement* Chi2ProbVsPhi;
	MonitorElement* Chi2ProbVsTheta;
	
	MonitorElement* DistanceOfClosestApproach;
	MonitorElement* DistanceOfClosestApproachToBS;
	MonitorElement* DistanceOfClosestApproachToPV;
	MonitorElement* DistanceOfClosestApproachVsTheta;
	MonitorElement* DistanceOfClosestApproachVsPhi;
	MonitorElement* DistanceOfClosestApproachToBSVsPhi;
	MonitorElement* DistanceOfClosestApproachToPVVsPhi;
	MonitorElement* DistanceOfClosestApproachVsEta;
	MonitorElement* xPointOfClosestApproach;
	MonitorElement* xPointOfClosestApproachToPV;
	MonitorElement* xPointOfClosestApproachVsZ0wrt000;
	MonitorElement* xPointOfClosestApproachVsZ0wrtBS;
	MonitorElement* xPointOfClosestApproachVsZ0wrtPV;
	MonitorElement* yPointOfClosestApproach;
	MonitorElement* yPointOfClosestApproachToPV;
	MonitorElement* yPointOfClosestApproachVsZ0wrt000;
	MonitorElement* yPointOfClosestApproachVsZ0wrtBS;
	MonitorElement* yPointOfClosestApproachVsZ0wrtPV;
	MonitorElement* zPointOfClosestApproach;
	MonitorElement* zPointOfClosestApproachToPV;
	MonitorElement* zPointOfClosestApproachVsPhi;
	MonitorElement* algorithm;
	// TESTING MEs
	MonitorElement* TESTDistanceOfClosestApproachToBS;
	MonitorElement* TESTDistanceOfClosestApproachToBSVsPhi;
	
	// add by Mia in order to deal w/ LS transitions
	MonitorElement* Chi2oNDF_lumiFlag;
	MonitorElement* NumberOfRecHitsPerTrack_lumiFlag;
	
	
	struct TkRecHitsPerSubDetMEs {
	  MonitorElement* NumberOfRecHitsPerTrack;
	  MonitorElement* NumberOfRecHitsPerTrackVsPhi;
	  MonitorElement* NumberOfRecHitsPerTrackVsEta;
	  MonitorElement* NumberOfLayersPerTrack;
	  MonitorElement* NumberOfLayersPerTrackVsPhi;
	  MonitorElement* NumberOfLayersPerTrackVsEta;
	  
	  int         detectorId;
	  std::string detectorTag;
	};
        std::map<std::string, TkRecHitsPerSubDetMEs> TkRecHitsPerSubDetMEMap;
	
	
	
        std::string histname;  //for naming the histograms according to algorithm used
};
#endif
