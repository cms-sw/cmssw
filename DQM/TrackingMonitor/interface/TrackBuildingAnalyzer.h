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
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

class TrackBuildingAnalyzer 
{
    public:
        using MVACollection = std::vector<float>;
        using QualityMaskCollection = std::vector<unsigned char>;

        TrackBuildingAnalyzer(const edm::ParameterSet&);
        ~TrackBuildingAnalyzer();
        void initHisto(DQMStore::IBooker & ibooker, const edm::ParameterSet&);
        void analyze
        (
            const edm::Event& iEvent, 
            const edm::EventSetup& iSetup, 
            const TrajectorySeed& seed, 
            const reco::BeamSpot& bs, 
            const edm::ESHandle<MagneticField>& theMF,
            const edm::ESHandle<TransientTrackingRecHitBuilder>& theTTRHBuilder
        );
        void analyze
        (
            const edm::Event& iEvent, 
            const edm::EventSetup& iSetup, 
            const TrackCandidate& candidate, 
            const reco::BeamSpot& bs, 
            const edm::ESHandle<MagneticField>& theMF,
            const edm::ESHandle<TransientTrackingRecHitBuilder>& theTTRHBuilder
        );
        void analyze
        (
            const edm::View<reco::Track>& trackCollection,
            const std::vector<const MVACollection *>& mvaCollections,
            const std::vector<const QualityMaskCollection *>& qualityMaskCollections
        );
        void analyze
        (
            const reco::CandidateView& regionCandidates
        );

    private:

        void fillHistos(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname);
        void bookHistos(std::string sname, DQMStore::IBooker & ibooker);

        // ----------member data ---------------------------

        // Candidates used for tracking regions
        MonitorElement* TrackingRegionCandidatePt;
        MonitorElement* TrackingRegionCandidateEta;
        MonitorElement* TrackingRegionCandidatePhi;
        MonitorElement* TrackingRegionCandidatePhiVsEta;

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

	std::vector<MonitorElement *> trackMVAs;
	std::vector<MonitorElement *> trackMVAsHP;
	std::vector<MonitorElement *> trackMVAsVsPtProfile;
	std::vector<MonitorElement *> trackMVAsHPVsPtProfile;
	std::vector<MonitorElement *> trackMVAsVsEtaProfile;
	std::vector<MonitorElement *> trackMVAsHPVsEtaProfile;
	
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
	bool doMVAPlots;
	bool doRegionPlots;
};
#endif
