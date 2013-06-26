#ifndef RecoMuon_MuonSeedGenerator_RPCCosmicSeedrecHitFinder_H
#define RecoMuon_MuonSeedGenerator_RPCCosmicSeedrecHitFinder_H

/** \class RPCSeedLayerFinder
 *  
 *   \author Haiyun.Teng - Peking University
 *
 *  
 */


#include "RecoMuon/MuonSeedGenerator/src/RPCSeedFinder.h"
#include <RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <FWCore/Framework/interface/EventSetup.h>

#ifndef RPCLayerNumber
#define RPCLayerNumber 12
#endif

#ifndef BarrelLayerNumber
#define BarrelLayerNumber 6
#endif

#ifndef EachEndcapLayerNumber
#define EachEndcapLayerNumber 3
#endif

class RPCCosmicSeedrecHitFinder {

    typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
    typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;

    public:
        RPCCosmicSeedrecHitFinder();
        ~RPCCosmicSeedrecHitFinder();
        void configure(const edm::ParameterSet& iConfig);
        void setEdge(const edm::EventSetup& iSetup);
        void unsetEdge();
        void setInput(MuonRecHitContainer (&recHits)[RPCLayerNumber]);
        void unsetInput();
        void setOutput(RPCSeedFinder *Seed); // Use the same RPCSeedFinder class
        void setLayers(const std::vector<unsigned int>& Layers);
        void fillrecHits();
    private:
        int LayerComponent();
        bool complete(const GlobalVector& lastSegment, const MuonRecHitPointer& lastrecHitRef);
        void checkandfill();
        bool isinsideAngleRange(const GlobalVector& lastSegment, const GlobalPoint& lastPosition, const GlobalPoint& currentPosition);
        bool iscorssEdge(const MuonRecHitPointer& lastrecHitRef, const MuonRecHitPointer& currentrecHitRef);
        bool isouterLayer(const MuonRecHitPointer& recHitRef);

        // ----------member data ---------------------------

        // parameters for configuration
        unsigned int BxRange;
        std::vector<int> ClusterSet;
        double MaxDeltaPhi;
        // Signal for call fillrecHits()
        bool isLayerset;
        bool isConfigured;
        bool isInputset;
        bool isOutputset;
        bool isEdgeset;
        // Signal for filling recHits
        bool isOuterLayerfilled;
        // Enable layers in Barrel and Endcap
        std::vector<unsigned int> LayersinRPC;
        // Data members
        std::vector<BoundPlane> innerBounds;
        bool isLayersmixed;
        MuonRecHitContainer AllrecHits[RPCLayerNumber];
        ConstMuonRecHitContainer therecHits;
        RPCSeedFinder *theSeed;
};

#endif
