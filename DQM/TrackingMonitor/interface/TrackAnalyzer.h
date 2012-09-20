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
// $Id: TrackAnalyzer.h,v 1.10 2011/07/18 14:32:47 fiori Exp $

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

        void doSoftReset(DQMStore * dqmStore_);
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
	bool doGoodTrackRecHitVsPhiVsEtaPerTrack_;

        MonitorElement* NumberOfRecHitsPerTrack;
        MonitorElement* NumberOfRecHitsFoundPerTrack;
        MonitorElement* NumberOfRecHitsLostPerTrack;
        MonitorElement* NumberOfLayersPerTrack;
        MonitorElement* NumberOfRecHitVsPhiVsEtaPerTrack;
        MonitorElement* GoodTrackNumberOfRecHitVsPhiVsEtaPerTrack;
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
        MonitorElement* xPointOfClosestApproachVsZ0;
        MonitorElement* yPointOfClosestApproach;
        MonitorElement* yPointOfClosestApproachVsZ0;
        MonitorElement* zPointOfClosestApproach;
	MonitorElement* algorithm;

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

        MonitorElement* GoodTrackChi2oNDF;
        MonitorElement* GoodTrackNumberOfRecHitsPerTrack;

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
