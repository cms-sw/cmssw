#ifndef RecoMuon_MuonSeedGenerator_RPCSeedOverlapper_H
#define RecoMuon_MuonSeedGenerator_RPCSeedOverlapper_H

/**  \class RPCSeedPattern
 *
 *  \author Haiyun.Teng - Peking University
 *
 *
 */


#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <DataFormats/Common/interface/OwnVector.h>
#include <DataFormats/TrackingRecHit/interface/TrackingRecHit.h>
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedPattern.h"

class RPCSeedOverlapper {

    typedef RPCSeedPattern::weightedTrajectorySeed weightedTrajectorySeed;

    public:
        RPCSeedOverlapper();
        ~RPCSeedOverlapper();
        void setIO(std::vector<weightedTrajectorySeed> *goodweightedRef, std::vector<weightedTrajectorySeed> *candidateweightedRef);
        void unsetIO();
        void run();    
        void configure(const edm::ParameterSet& iConfig);
        void setEventSetup(const edm::EventSetup& iSetup);
    private:
        void CheckOverlap(const edm::EventSetup& iSetup, std::vector<weightedTrajectorySeed> *SeedsRef);
        bool isShareHit(const edm::OwnVector<TrackingRecHit> &RecHits, const TrackingRecHit& hit, edm::ESHandle<RPCGeometry> rpcGeometry);
        // Signal for call run()
        bool isConfigured;
        bool isIOset;
        bool isEventSetupset;
        // Parameters for configuration
        bool isCheckgoodOverlap;
        bool isCheckcandidateOverlap;
        unsigned int ShareRecHitsNumberThreshold;
        // IO ref
        std::vector<weightedTrajectorySeed> *goodweightedSeedsRef;
        std::vector<weightedTrajectorySeed> *candidateweightedSeedsRef;
        const edm::EventSetup *eSetup;
};

#endif
