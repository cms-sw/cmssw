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
// $Id: TrackAnalyzer.h,v 1.16 2012/04/27 15:56:48 tosi Exp $

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class DQMStore;

class TrackAnalyzer 
{
    public:
        TrackAnalyzer(const edm::ParameterSet&);
        virtual ~TrackAnalyzer();
        virtual void beginJob(DQMStore * dqmStore_);

        virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track);

        void doSoftReset  (DQMStore * dqmStore_);
        void doReset      (DQMStore * dqmStore_);
        void undoSoftReset(DQMStore * dqmStore_);
        void setLumiFlag();

    private:

        void fillHistosForState(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname);
        void bookHistosForState(std::string sname, DQMStore * dqmStore_);
        void doTrackerSpecificInitialization(DQMStore * dqmStore_);
        void doTrackerSpecificFillHists(const reco::Track & track);

        // ----------member data ---------------------------

        edm::ParameterSet conf_;

        bool doTrackerSpecific_;
        bool doAllPlots_;
        bool doBSPlots_;
	bool doGoodTrackPlots_;
	bool doDCAPlots_;
	bool doGeneralPropertiesPlots_;
	bool doMeasurementStatePlots_;
	bool doHitPropertiesPlots_;
	bool doRecHitVsPhiVsEtaPerTrack_;
	// ADD by Mia
	bool doLayersVsPhiVsEtaPerTrack_;
	bool doGoodTrackRecHitVsPhiVsEtaPerTrack_;
	bool doGoodTrackLayersVsPhiVsEtaPerTrack_;
	bool doGoodTrack2DChi2Plots_;

	// ADD by Mia in order to clean the tracking MEs
	// do not plot *Theta* and TrackPx* and TrackPy*
	bool doThetaPlots_;
	bool doTrackPxPyPlots_;
	// ADD by Mia in order to not plot DistanceOfClosestApproach w.r.t. (0,0,0)
	// the DistanceOfClosestApproach w.r.t. the beam-spot is already shown in DistanceOfClosestApproachToBS
	bool doDCAwrt000Plots_;

	bool doLumiAnalysis_;

	// ADD by Mia in order to turnON test MEs
	bool doTestPlots_;

        MonitorElement* NumberOfRecHitsPerTrack;
        MonitorElement* NumberOfRecHitsFoundPerTrack;
        MonitorElement* NumberOfRecHitsLostPerTrack;
        MonitorElement* NumberOfLayersPerTrack;
        MonitorElement* NumberOfRecHitVsPhiVsEtaPerTrack;
        MonitorElement* NumberOfLayersVsPhiVsEtaPerTrack;
        MonitorElement* Chi2;
        MonitorElement* Chi2Prob;
        MonitorElement* Chi2oNDF;
        MonitorElement* DistanceOfClosestApproach;
        MonitorElement* DistanceOfClosestApproachToBS;
        MonitorElement* DistanceOfClosestApproachVsTheta;
        MonitorElement* DistanceOfClosestApproachVsPhi;
        MonitorElement* DistanceOfClosestApproachToBSVsPhi;
        MonitorElement* DistanceOfClosestApproachVsEta;
        MonitorElement* xPointOfClosestApproach;
        MonitorElement* xPointOfClosestApproachVsZ0wrt000;
        MonitorElement* xPointOfClosestApproachVsZ0wrtBS;
        MonitorElement* yPointOfClosestApproach;
        MonitorElement* yPointOfClosestApproachVsZ0wrt000;
        MonitorElement* yPointOfClosestApproachVsZ0wrtBS;
        MonitorElement* zPointOfClosestApproach;
        MonitorElement* zPointOfClosestApproachVsPhi;
	MonitorElement* algorithm;
	// TESTING MEs
        MonitorElement* TESTDistanceOfClosestApproachToBS;
        MonitorElement* TESTDistanceOfClosestApproachToBSVsPhi;

	// add by Mia in order to deal w/ LS transitions
	MonitorElement* Chi2oNDF_lumiFlag;
	MonitorElement* NumberOfRecHitsPerTrack_lumiFlag;
	MonitorElement* GoodTrackChi2oNDF_lumiFlag;
	MonitorElement* GoodTrackNumberOfRecHitsPerTrack_lumiFlag;

        MonitorElement* NumberOfTOBRecHitsPerTrack;
        MonitorElement* NumberOfTOBRecHitsPerTrackVsPhiProfile;
        MonitorElement* NumberOfTOBRecHitsPerTrackVsEtaProfile;
        MonitorElement* NumberOfTOBLayersPerTrack;
        MonitorElement* NumberOfTOBLayersPerTrackVsPhiProfile;
        MonitorElement* NumberOfTOBLayersPerTrackVsEtaProfile;

	MonitorElement* NumberOfTIBRecHitsPerTrack;
        MonitorElement* NumberOfTIBRecHitsPerTrackVsPhiProfile;
        MonitorElement* NumberOfTIBRecHitsPerTrackVsEtaProfile;
	MonitorElement* NumberOfTIBLayersPerTrack;
        MonitorElement* NumberOfTIBLayersPerTrackVsPhiProfile;
        MonitorElement* NumberOfTIBLayersPerTrackVsEtaProfile;

	MonitorElement* NumberOfTIDRecHitsPerTrack;
        MonitorElement* NumberOfTIDRecHitsPerTrackVsPhiProfile;
        MonitorElement* NumberOfTIDRecHitsPerTrackVsEtaProfile;
	MonitorElement* NumberOfTIDLayersPerTrack;
        MonitorElement* NumberOfTIDLayersPerTrackVsPhiProfile;
        MonitorElement* NumberOfTIDLayersPerTrackVsEtaProfile;

        MonitorElement* NumberOfTECRecHitsPerTrack;
        MonitorElement* NumberOfTECRecHitsPerTrackVsPhiProfile;
        MonitorElement* NumberOfTECRecHitsPerTrackVsEtaProfile;
	MonitorElement* NumberOfTECLayersPerTrack;
        MonitorElement* NumberOfTECLayersPerTrackVsPhiProfile;
        MonitorElement* NumberOfTECLayersPerTrackVsEtaProfile;

	MonitorElement* NumberOfPixBarrelRecHitsPerTrack;
        MonitorElement* NumberOfPixBarrelRecHitsPerTrackVsPhiProfile;
        MonitorElement* NumberOfPixBarrelRecHitsPerTrackVsEtaProfile;
        MonitorElement* NumberOfPixBarrelLayersPerTrack;
        MonitorElement* NumberOfPixBarrelLayersPerTrackVsPhiProfile;
        MonitorElement* NumberOfPixBarrelLayersPerTrackVsEtaProfile;

	MonitorElement* NumberOfPixEndcapRecHitsPerTrack;
        MonitorElement* NumberOfPixEndcapRecHitsPerTrackVsPhiProfile;
        MonitorElement* NumberOfPixEndcapRecHitsPerTrackVsEtaProfile;
        MonitorElement* NumberOfPixEndcapLayersPerTrack;
        MonitorElement* NumberOfPixEndcapLayersPerTrackVsPhiProfile;
        MonitorElement* NumberOfPixEndcapLayersPerTrackVsEtaProfile;

        MonitorElement* GoodTrackNumberOfRecHitVsPhiVsEtaPerTrack;
        MonitorElement* GoodTrackNumberOfLayersVsPhiVsEtaPerTrack;
        MonitorElement* GoodTrackNumberOfRecHitsPerTrackVsPhiProfile;
        MonitorElement* GoodTrackNumberOfRecHitsPerTrackVsEtaProfile;
        MonitorElement* GoodTrackNumberOfFoundRecHitsPerTrackVsPhiProfile;
        MonitorElement* GoodTrackNumberOfFoundRecHitsPerTrackVsEtaProfile;
        MonitorElement* GoodTrackChi2oNDF;
        MonitorElement* GoodTrackChi2Prob;
        MonitorElement* GoodTrackChi2oNDFVsPhi;
        MonitorElement* GoodTrackChi2ProbVsPhi;
        MonitorElement* GoodTrackChi2oNDFVsEta;
        MonitorElement* GoodTrackChi2ProbVsEta;
        MonitorElement* GoodTrackNumberOfRecHitsPerTrack;
        MonitorElement* GoodTrackNumberOfFoundRecHitsPerTrack;
	MonitorElement* GoodTrackAlgorithm;

        struct TkParameterMEs 
        {
            TkParameterMEs()
                : TrackP(NULL)
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
                , NumberOfRecHitsPerTrackVsPhiProfile(NULL)
                , NumberOfRecHitsPerTrackVsThetaProfile(NULL)
                , NumberOfRecHitsPerTrackVsEtaProfile(NULL)
                , NumberOfLayersPerTrackVsPhi(NULL)
                , NumberOfLayersPerTrackVsTheta(NULL)
                , NumberOfLayersPerTrackVsEta(NULL)
                , NumberOfLayersPerTrackVsPhiProfile(NULL)
                , NumberOfLayersPerTrackVsThetaProfile(NULL)
                , NumberOfLayersPerTrackVsEtaProfile(NULL)

                , Chi2oNDFVsTheta(NULL)
                , Chi2oNDFVsPhi(NULL)
                , Chi2oNDFVsEta(NULL)
                , Chi2oNDFVsThetaProfile(NULL)
                , Chi2oNDFVsPhiProfile(NULL)
                , Chi2oNDFVsEtaProfile(NULL)

		, GoodTrackPt(NULL)
		, GoodTrackEta(NULL)
		, GoodTrackPhi(NULL)
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
            MonitorElement* NumberOfRecHitsPerTrackVsPhiProfile;
            MonitorElement* NumberOfRecHitsPerTrackVsThetaProfile;
            MonitorElement* NumberOfRecHitsPerTrackVsEtaProfile;
            MonitorElement* NumberOfLayersPerTrackVsPhi;
            MonitorElement* NumberOfLayersPerTrackVsTheta;
            MonitorElement* NumberOfLayersPerTrackVsEta;
            MonitorElement* NumberOfLayersPerTrackVsPhiProfile;
            MonitorElement* NumberOfLayersPerTrackVsThetaProfile;
            MonitorElement* NumberOfLayersPerTrackVsEtaProfile;

            MonitorElement* Chi2oNDFVsTheta;
            MonitorElement* Chi2oNDFVsPhi;
            MonitorElement* Chi2oNDFVsEta;
            MonitorElement* Chi2oNDFVsThetaProfile;
            MonitorElement* Chi2oNDFVsPhiProfile;
            MonitorElement* Chi2oNDFVsEtaProfile;

	    MonitorElement* GoodTrackPt;
	    MonitorElement* GoodTrackEta;
	    MonitorElement* GoodTrackPhi;
        };

        std::map<std::string, TkParameterMEs> TkParameterMEMap;

        std::string histname;  //for naming the histograms according to algorithm used
};
#endif
