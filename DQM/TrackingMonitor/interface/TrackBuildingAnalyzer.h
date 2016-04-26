#ifndef TrackBuildingAnalyzer_H
#define TrackBuildingAnalyzer_H
// -*- C++ -*-
//
// 
/**\class TrackBuildingAnalyzer TrackBuildingAnalyzer.cc 
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Ryan Kelley 
//         Created:  Sat 28 13;30:00 CEST 2009
//

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"


class DQMStore;

class TrackBuildingAnalyzer 
{
    public:
        TrackBuildingAnalyzer(const edm::ParameterSet&);
        virtual ~TrackBuildingAnalyzer();
        virtual void initHisto(DQMStore::IBooker & ibooker);
        virtual void analyze
        (
            const edm::Event& iEvent, 
            const edm::EventSetup& iSetup, 
            const TrajectorySeed& seed, 
            const reco::BeamSpot& bs, 
            const edm::ESHandle<MagneticField>& theMF,
            const edm::ESHandle<TransientTrackingRecHitBuilder>& theTTRHBuilder
        );
        virtual void analyze
        (
            const edm::Event& iEvent, 
            const edm::EventSetup& iSetup, 
            const TrackCandidate& candidate, 
            const reco::BeamSpot& bs, 
            const edm::ESHandle<MagneticField>& theMF,
            const edm::ESHandle<TransientTrackingRecHitBuilder>& theTTRHBuilder
        );

    private:

        void fillHistos(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname);
        void bookHistos(std::string sname, DQMStore::IBooker & ibooker);

        // ----------member data ---------------------------

        edm::ParameterSet conf_;

        // Track Seeds
        MonitorElement* SeedPt;
        MonitorElement* SeedEta;
        MonitorElement* SeedPhi;
	MonitorElement* SeedPhiVsEta;
        MonitorElement* SeedTheta;
        MonitorElement* SeedQ;
        MonitorElement* SeedDxy;
        MonitorElement* SeedDz;
        MonitorElement* NumberOfRecHitsPerSeed;
        MonitorElement* NumberOfRecHitsPerSeedVsPhiProfile;
        MonitorElement* NumberOfRecHitsPerSeedVsEtaProfile;

        // Track Candidate
        MonitorElement* TrackCandPt;
        MonitorElement* TrackCandEta;
        MonitorElement* TrackCandPhi;
        MonitorElement* TrackCandPhiVsEta;
	MonitorElement* TrackCandTheta;
        MonitorElement* TrackCandQ;
        MonitorElement* TrackCandDxy;
        MonitorElement* TrackCandDz;
        MonitorElement* NumberOfRecHitsPerTrackCand;
        MonitorElement* NumberOfRecHitsPerTrackCandVsPhiProfile;
        MonitorElement* NumberOfRecHitsPerTrackCandVsEtaProfile;

	MonitorElement* stoppingSource;
	MonitorElement* stoppingSourceVSeta;
	MonitorElement* stoppingSourceVSphi;
	
        std::string histname;  //for naming the histograms according to algorithm used

	//to disable some plots
	bool doAllPlots;
	bool doAllSeedPlots;
	bool doTCPlots;
	bool doAllTCPlots;
       	bool doPT;
	bool doETA;
	bool doPHI;
	bool doPHIVsETA;
	bool doTheta;
	bool doQ;
	bool doDxy;
	bool doDz;
	bool doNRecHits;
	bool doProfPHI;
	bool doProfETA;
	bool doStopSource;
};
#endif
