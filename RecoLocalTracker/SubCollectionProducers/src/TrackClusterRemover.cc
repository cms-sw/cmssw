#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
//
// class decleration
//

class TrackClusterRemover : public edm::EDProducer {
    public:
        TrackClusterRemover(const edm::ParameterSet& iConfig) ;
        ~TrackClusterRemover() ;
        void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;
    private:
        struct ParamBlock {
            ParamBlock() : isSet_(false), usesCharge_(false) {}
            ParamBlock(const edm::ParameterSet& iConfig) :
                isSet_(true), 
                usesCharge_(iConfig.exists("maxCharge")),
                usesSize_(iConfig.exists("maxSize")),
                maxChi2_(iConfig.getParameter<double>("maxChi2")),
                maxCharge_(usesCharge_ ? iConfig.getParameter<double>("maxCharge") : 0), 
                maxSize_(usesSize_ ? iConfig.getParameter<uint32_t>("maxSize") : 0) { }
            bool  isSet_, usesCharge_, usesSize_;
            float maxChi2_, maxCharge_;
            size_t maxSize_;
        };
        static const unsigned int NumberOfParamBlocks = 6;

        edm::InputTag trajectories_, stripClusters_, pixelClusters_;
        bool mergeOld_;
        edm::InputTag oldRemovalInfo_;

        ParamBlock pblocks_[NumberOfParamBlocks];
        void readPSet(const edm::ParameterSet& iConfig, const std::string &name, 
                int id1=-1, int id2=-1, int id3=-1, int id4=-1, int id5=-1, int id6=-1) ;

        std::vector<uint64_t> pixels, strips;    // avoid unneed alloc/dealloc of this
        edm::ProductID pixelSourceProdID, stripSourceProdID; // ProdIDs refs must point to (for consistency tests)

        inline void process(const TrackingRecHit *hit, float chi2);
        inline void process(const SiStripRecHit2D *hit2D, uint32_t subdet);

        inline static uint32_t detid(const uint64_t &x) { return static_cast<uint32_t>(x >> 32); }
        inline static uint32_t index(const uint64_t &x) { return static_cast<uint32_t>(x & 0xFFFFFFFF); }
        inline static uint64_t pack (const int32_t &detid, const uint32_t &idx) { return ((static_cast<uint64_t>(detid)) << 32) + idx; }
        inline static uint64_t pack (const  std::pair<edm::det_id_type,size_t> &pair) { return ((static_cast<uint64_t>(pair.first)) << 32) + pair.second; }

        template<typename T> 
        std::auto_ptr<edm::DetSetVector<T> >
        cleanup(const edm::DetSetVector<T> &oldClusters, const std::vector<uint64_t> &usedOnes, 
                    edmNew::DetSetVector<uint16_t> &refs, const edmNew::DetSetVector<uint16_t> *oldRefs) ;

        // Carries in full removal info about a given det from oldRefs
        void mergeFull(   edmNew::DetSetVector<uint16_t> &refs, const edmNew::DetSetVector<uint16_t> &oldRefs, uint32_t detid) ;
        // Merges in information about a given det between oldRefs and the outcome of the current removal run. 
        // if oldRefs knows nothing about that det, it just calls mergeNew
        void mergePartial(edmNew::DetSetVector<uint16_t> &refs, const edmNew::DetSetVector<uint16_t> &oldRefs, uint32_t detid,
                            uint16_t newSize, const std::vector<uint8_t> &good) ; 
        // Writes new removal info about a given det, using the outcome of the current removal run.
        void mergeNew(    edmNew::DetSetVector<uint16_t> &refs,                                                uint32_t detid,
                            uint16_t newSize, const std::vector<uint8_t> &good) ; 
};


using namespace std;
using namespace edm;
using namespace reco;

void
TrackClusterRemover::readPSet(const edm::ParameterSet& iConfig, const std::string &name, 
    int id1, int id2, int id3, int id4, int id5, int id6) 
{
    if (iConfig.exists(name)) {
        ParamBlock pblock(iConfig.getParameter<ParameterSet>(name));
        if (id1 == -1) {
            fill(pblocks_, pblocks_+NumberOfParamBlocks, pblock);
        } else {
            pblocks_[id1] = pblock;
            if (id2 != -1) pblocks_[id2] = pblock;
            if (id3 != -1) pblocks_[id3] = pblock;
            if (id4 != -1) pblocks_[id4] = pblock;
            if (id5 != -1) pblocks_[id5] = pblock;
            if (id6 != -1) pblocks_[id6] = pblock;
        }
    }
}

TrackClusterRemover::TrackClusterRemover(const ParameterSet& iConfig):
    trajectories_(iConfig.getParameter<InputTag>("trajectories")),
    stripClusters_(iConfig.getParameter<InputTag>("stripClusters")),
    pixelClusters_(iConfig.getParameter<InputTag>("pixelClusters")),
    mergeOld_(iConfig.exists("oldClusterRemovalInfo")),
    oldRemovalInfo_(mergeOld_ ? iConfig.getParameter<InputTag>("oldClusterRemovalInfo") : InputTag("NONE"))
{
    produces< DetSetVector<SiPixelCluster> >();
    produces< DetSetVector<SiStripCluster> >();
    produces< ClusterRemovalInfo >();

    fill(pblocks_, pblocks_+NumberOfParamBlocks, ParamBlock());
    readPSet(iConfig, "Common",-1);
    readPSet(iConfig, "Pixel" ,0,1);
    readPSet(iConfig, "PXB" ,0);
    readPSet(iConfig, "PXE" ,1);
    readPSet(iConfig, "Strip" ,2,3,4,5);
    readPSet(iConfig, "StripInner" ,2,3);
    readPSet(iConfig, "StripOuter" ,4,5);
    readPSet(iConfig, "TIB" ,2);
    readPSet(iConfig, "TID" ,3);
    readPSet(iConfig, "TOB" ,4);
    readPSet(iConfig, "TEC" ,5);

    bool usingCharge = false;
    for (size_t i = 0; i < NumberOfParamBlocks; ++i) {
        if (!pblocks_[i].isSet_) throw cms::Exception("Configuration Error") << "TrackClusterRemover: Missing configuration for detector with subDetID = " << (i+1);
        if (pblocks_[i].usesCharge_ && !usingCharge) {
            throw cms::Exception("Configuration Error") << "TrackClusterRemover: Configuration for subDetID = " << (i+1) << " uses cluster charge, which is not enabled.";
        }
    }
}


TrackClusterRemover::~TrackClusterRemover()
{
}

void TrackClusterRemover::mergeFull(edmNew::DetSetVector<uint16_t> &refs,
                                            const edmNew::DetSetVector<uint16_t> &oldRefs,
                                            uint32_t detid) 
{
    // the new is full, so we just have to convert
    typedef edmNew::DetSetVector<uint16_t>::FastFiller  RefDS;
    typedef edmNew::DetSetVector<uint16_t>::const_iterator RefMatch;
    typedef edmNew::DetSet<uint16_t> OldRefDS;
    RefMatch match = oldRefs.find(detid);
    if (match != oldRefs.end()) {
        OldRefDS olds = *match;
        RefDS merged(refs, detid);
        merged.resize(olds.size());
        std::copy(olds.begin(), olds.end(), merged.begin());
    } 
}

void TrackClusterRemover::mergeNew(edmNew::DetSetVector<uint16_t> &refs,
                                            uint32_t detid,
                                            uint16_t newSize,
                                            const vector<uint8_t> &good)
{
    typedef edmNew::DetSetVector<uint16_t>::FastFiller  RefDS;
        RefDS merged(refs, detid);
        merged.resize(newSize);
        vector<uint8_t>::const_iterator it = good.begin(), ed = good.end();
        uint16_t iold, inew;
        for (iold = inew = 0; it != ed; ++it, ++iold) {
            if (*it) { 
                merged[inew] = iold; 
                ++inew; 
            }
        }
}

                                            
void TrackClusterRemover::mergePartial(edmNew::DetSetVector<uint16_t> &refs,
                                            const edmNew::DetSetVector<uint16_t> &oldRefs,
                                            uint32_t detid,
                                            uint16_t newSize,
                                            const vector<uint8_t> &good)
{
    typedef edmNew::DetSetVector<uint16_t>::FastFiller  RefDS;
    typedef edmNew::DetSetVector<uint16_t>::const_iterator RefMatch;
    typedef edmNew::DetSet<uint16_t> OldRefDS;
    RefMatch match = oldRefs.find(detid);
    if (match != oldRefs.end()) {
        OldRefDS olds = *match;
        if (newSize > olds.size()) {
                throw cms::Exception("Inconsistent Data") <<  
                    "The ClusterRemovalInfo passed to TrackClusterRemover does not match with the clusters!";
        }
        RefDS merged(refs, detid);
        merged.resize(newSize);
        vector<uint8_t>::const_iterator it = good.begin(), ed = good.end();
        uint16_t iold, inew;
        for (iold = inew = 0; it != ed; ++it, ++iold) {
            if (*it) { merged[inew] = olds[iold]; ++inew; }
        }
   } else {
        mergeNew(refs,detid,newSize,good);
    }
}
 
template<typename T> 
std::auto_ptr<edm::DetSetVector<T> >
TrackClusterRemover::cleanup(const edm::DetSetVector<T> &oldClusters, const std::vector<uint64_t> &usedOnes, 
                                edmNew::DetSetVector<uint16_t> &refs, const edmNew::DetSetVector<uint16_t> *oldRefs) {  
    typedef typename edmNew::DetSetVector<uint16_t>::FastFiller  RefDS;
    // room for output 
    std::vector<DetSet<T> > work;
    work.reserve(oldClusters.size());

    // start from the beginning
    typename vector<uint64_t>::const_iterator used = usedOnes.begin(); 
    uint32_t used_detid = detid(*used);

    // cluster removal loop
    for (typename DetSetVector<T>::const_iterator itdet = oldClusters.begin(), enddet = oldClusters.end(); itdet != enddet; ++itdet) {
        const DetSet<T> &oldDS = *itdet;

        if (oldDS.empty()) continue; // skip empty detsets

        uint32_t id = oldDS.detId();

        // look in used dets until I find this detid
        if (id > used_detid) {
            do { ++used; } while ( *used < pack(id,0) );
            used_detid = detid(*used);
        }
        
        if (id < used_detid) {         // the detid is not there
            work.push_back(oldDS);     // keep the whole stuff
            if (oldRefs) mergeFull(refs, *oldRefs, id);
//DBG//     for (size_t idx = 0; idx < oldDS.size(); ++idx) {
//DBG//             cout << "\tNEW " << (DetId(id).subdetId() <= 2 ? "PIXEL" : "STRIP") 
//DBG//                         << " [ " << id  << " / " << idx << " ] OK" << endl;
//DBG//     }
        } else {
            // qui cominciano le dolenti note a farsi sentire

            // count how many hits have been taken
            uint32_t oldSize = oldDS.size(), taken = 0;
            vector<uint8_t> good(oldSize, true); // uint_8 faster than bool, and vector is small
            do {
                uint32_t idx = index(*used);
                if (good[idx]) { taken++; good[idx] = false; }
                ++used;
            } while (detid(*used) ==  used_detid);
            used_detid = detid(*used); // important            

            // if not fully depleted, I have to write it
            if (taken < oldSize) {
                size_t newSize = oldSize - taken;

                // output container
                DetSet<T> newDS(id); 
                newDS.reserve(newSize);

                typename DetSet<T>::const_iterator itold = oldDS.begin();
                uint16_t idx = 0, idxnew = 0;
                do {
//DBG//             cout << "\tNEW " << (DetId(id).subdetId() <= 2 ? "PIXEL" : "STRIP") 
//DBG//                         << " [ " << id  << " / " << idx << " ] " 
//DBG//                                  << ( good[idx] ? "OK" : "NO" ) << endl;
                    if (good[idx]) {
                        newDS.push_back(*itold);
                        ++idxnew;
                    }
                    ++idx; ++itold;
                } while (idx < oldSize);
                work.push_back(newDS);

                if (oldRefs) {
                    mergePartial(refs, *oldRefs, id, newSize, good);
                } else {
                    mergeNew(refs, id, newSize, good);
                }
            } // if not fully depleted
//DBG//     else {
//DBG//         for (size_t idx = 0; idx < oldDS.size(); ++idx) {
//DBG//                 cout << "\tNEW " << (DetId(id).subdetId() <= 2 ? "PIXEL" : "STRIP") 
//DBG//                             << " [ " << id  << " / " << idx << " ] NO" << endl;
//DBG//         }
//DBG//     }
        } // work on current det
    } // loop on dets

    //auto_ptr<DetSetVector<T> > newClusters(new DetSetVector<T>(work)); // FIXME: use the line below as soon as DSV supports it
    auto_ptr<DetSetVector<T> > newClusters(new DetSetVector<T>(work, true)); // already sorted
    return newClusters;
}

void TrackClusterRemover::process(const SiStripRecHit2D *hit, uint32_t subdet) {
    SiStripRecHit2D::ClusterRef cluster = hit->cluster();

    if (cluster.id() != stripSourceProdID) throw cms::Exception("Inconsistent Data") << 
            "TrackClusterRemover: strip cluster ref from Product ID = " << cluster.id() << 
            " does not match with source cluster collection (ID = " << stripSourceProdID << ")\n.";

    assert(cluster.id() == stripSourceProdID);
    if (pblocks_[subdet-1].usesSize_ && (cluster->amplitudes().size() > pblocks_[subdet-1].maxSize_)) return;

//DBG// cout << "Individual HIT " << cluster.key().first << ", INDEX = " << cluster.key().second << endl;
    strips.push_back(pack(cluster.key()));
}

void TrackClusterRemover::process(const TrackingRecHit *hit, float chi2) {
    DetId detid = hit->geographicalId(); 
    uint32_t subdet = detid.subdetId();

    assert ((subdet > 0) && (subdet <= NumberOfParamBlocks));

    // chi2 cut
    if (chi2 > pblocks_[subdet-1].maxChi2_) return;

    if ((subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap)) {
        // this is a pixel, and i *know* it is
        const SiPixelRecHit *pixelHit = static_cast<const SiPixelRecHit *>(hit);

        SiPixelRecHit::ClusterRef cluster = pixelHit->cluster();

        if (cluster.id() != pixelSourceProdID) throw cms::Exception("Inconsistent Data") << 
                "TrackClusterRemover: pixel cluster ref from Product ID = " << cluster.id() << 
                " does not match with source cluster collection (ID = " << pixelSourceProdID << ")\n.";

        assert(cluster.id() == pixelSourceProdID);
//DBG// cout << "HIT NEW PIXEL DETID = " << detid.rawId() << ", Cluster [ " << cluster.key().first << " / " <<  cluster.key().second << " ] " << endl;

        // if requested, cut on cluster size
        if (pblocks_[subdet-1].usesSize_ && (cluster->pixels().size() > pblocks_[subdet-1].maxSize_)) return;

        // mark as used
        pixels.push_back(pack(cluster.key()));
    } else { // aka Strip
#define HIT_SPLITTING_TYPEID
#if defined(ASSUME_HIT_SPLITTING)
        const SiStripRecHit2D *stripHit = static_cast<const SiStripRecHit2D *>(hit);
        process(stripHit,subdet);
#elif defined(HIT_SPLITTING_TYPEID)
        const type_info &hitType = typeid(*hit);
        if (hitType == typeid(SiStripRecHit2D)) {
            const SiStripRecHit2D *stripHit = static_cast<const SiStripRecHit2D *>(hit);
//DBG//     cout << "Plain RecHit 2D: " << endl;
            process(stripHit,subdet);
        } else if (hitType == typeid(SiStripMatchedRecHit2D)) {
            const SiStripMatchedRecHit2D *matchHit = static_cast<const SiStripMatchedRecHit2D *>(hit);
//DBG//     cout << "Matched RecHit 2D: " << endl;
            process(matchHit->monoHit(),subdet);
            process(matchHit->stereoHit(),subdet);
        } else if (hitType == typeid(ProjectedSiStripRecHit2D)) {
            const ProjectedSiStripRecHit2D *projHit = static_cast<const ProjectedSiStripRecHit2D *>(hit);
//DBG//     cout << "Projected RecHit 2D: " << endl;
            process(&projHit->originalHit(),subdet);
        } else throw cms::Exception("NOT IMPLEMENTED") << "Don't know how to handle " << hitType.name() << " on detid " << detid.rawId() << "\n";
#else 
        //TODO: optimize using layer info to do the correct dyn_cast ?
        if (const SiStripRecHit2D *stripHit = dynamic_cast<const SiStripRecHit2D *>(hit)) {
//DBG//     cout << "Plain RecHit 2D: " << endl;
            process(stripHit,subdet);
        } else if (const SiStripMatchedRecHit2D *matchHit = dynamic_cast<const SiStripMatchedRecHit2D *>(hit)) {
//DBG//     cout << "Matched RecHit 2D: " << endl;
            process(matchHit->monoHit(),subdet);
            process(matchHit->stereoHit(),subdet);
        } else if (const ProjectedSiStripRecHit2D *projHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(hit)) {
//DBG//     cout << "Projected RecHit 2D: " << endl;
            process(&projHit->originalHit(),subdet);
        } else throw cms::Exception("NOT IMPLEMENTED") << "Don't know how to handle " << typeid(*hit).name() << " on detid " << detid.rawId() << "\n";
#endif
    }
}

/*   Schematic picture of n-th step Iterative removal
 *   (that os removing clusters after n-th step tracking)
 *   clusters:    [ C1 ] -> [ C2 ] -> ... -> [ Cn ] -> [ Cn + 1 ]
 *                   ^                          ^           ^--- OUTPUT "new" ID
 *                   |-- before any removal     |----- Source clusters                                                                             
 *                   |-- OUTPUT "old" ID        |----- Hits in Traj. point here                       
 *                   |                          \----- Old ClusterRemovalInfo "new" ID                 
 *                   \-- Old ClusterRemovalInfo "old" ID                                                  
 */


void
TrackClusterRemover::produce(Event& iEvent, const EventSetup& iSetup)
{
    ProductID pixelOldProdID, stripOldProdID;

    Handle<DetSetVector<SiPixelCluster> > pixelClusters;
    iEvent.getByLabel(pixelClusters_, pixelClusters);
    pixelSourceProdID = pixelClusters.id();

    Handle<DetSetVector<SiStripCluster> > stripClusters;
    iEvent.getByLabel(stripClusters_, stripClusters);
    stripSourceProdID = stripClusters.id();

    Handle<ClusterRemovalInfo> oldRemovalInfo;
    if (mergeOld_) { 
        iEvent.getByLabel(oldRemovalInfo_, oldRemovalInfo); 
        // Check ProductIDs
        if ( (oldRemovalInfo->stripNewProdID() == stripClusters.id()) &&
             (oldRemovalInfo->pixelNewProdID() == pixelClusters.id()) ) {

            pixelOldProdID = oldRemovalInfo->pixelProdID();
            stripOldProdID = oldRemovalInfo->stripProdID();

            // if Hits in the Trajectory are already rekeyed you need the following
            //    pixelSourceProdID = oldRemovalInfo->pixelProdID();
            //    stripSourceProdID = oldRemovalInfo->stripProdID();
            // but also some other parts of the index logic must be changed
        } else {
            throw cms::Exception("Inconsistent Data") << "TrackClusterRemover: " <<
                "Input collection product IDs are [pixel: " << pixelClusters.id() << ", strip: " << stripClusters.id() << "] \n" <<
                "\t but the *old* ClusterRemovalInfo " << oldRemovalInfo_.encode() << " refers as 'new product ids' to " <<
                    "[pixel: " << oldRemovalInfo->pixelNewProdID() << ", strip: " << oldRemovalInfo->stripNewProdID() << "]\n" << 
                "NOTA BENE: when running TrackClusterRemover with an old ClusterRemovalInfo the hits in the trajectory MUST be already re-keyed.\n";
        }
    } else { // then Old == Source
        pixelOldProdID = pixelSourceProdID;
        stripOldProdID = stripSourceProdID;
    }

    //Handle<TrajTrackAssociationCollection> trajectories; 
    Handle<vector<Trajectory> > trajectories; 
    iEvent.getByLabel(trajectories_, trajectories);

    strips.clear(); pixels.clear();

    //for (TrajTrackAssociationCollection::const_iterator it = trajectories->begin(), ed = trajectories->end(); it != ed; ++it)  {
    //    const Trajectory &tj = * it->key;
    for (vector<Trajectory>::const_iterator it = trajectories->begin(), ed = trajectories->end(); it != ed; ++it)  {
        const Trajectory &tj = * it;
        const vector<TrajectoryMeasurement> &tms = tj.measurements();
        vector<TrajectoryMeasurement>::const_iterator itm, endtm;
        for (itm = tms.begin(), endtm = tms.end(); itm != endtm; ++itm) {
            const TrackingRecHit *hit = itm->recHit()->hit();
            if (!hit->isValid()) continue; 
            process( hit, itm->estimate() );
        }
    }
    sort(pixels.begin(), pixels.end());
    sort(strips.begin(), strips.end());
    pixels.push_back(pack(0xFFFFFFFF,0xFFFFFFFF)); // as end marker
    strips.push_back(pack(0xFFFFFFFF,0xFFFFFFFF)); // as end marker

    auto_ptr<ClusterRemovalInfo> cri(new ClusterRemovalInfo(pixelOldProdID, stripOldProdID));
    auto_ptr<DetSetVector<SiPixelCluster> > newPixelClusters = cleanup(*pixelClusters, pixels, 
                cri->pixelIndices(), mergeOld_ ? &oldRemovalInfo->pixelIndices() : 0);
    auto_ptr<DetSetVector<SiStripCluster> > newStripClusters = cleanup(*stripClusters, strips, 
                cri->stripIndices(), mergeOld_ ? &oldRemovalInfo->stripIndices() : 0);

    OrphanHandle<DetSetVector<SiPixelCluster> > newPixels = iEvent.put(newPixelClusters); 
    cri->setNewPixelProdID(newPixels.id());
    OrphanHandle<DetSetVector<SiStripCluster> > newStrips = iEvent.put(newStripClusters); 
    cri->setNewStripProdID(newStrips.id());

//DBG// cout << "Source pixels = " << cri->pixelProdID() << endl;
//DBG// cout << "Source strips = " << cri->stripProdID() << endl;
//DBG// cout << "Destination pixels = " << cri->pixelNewProdID() << endl;
//DBG// cout << "Destination strips = " << cri->stripNewProdID() << endl;

    iEvent.put(cri);

    pixels.clear(); strips.clear(); 
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_ANOTHER_FWK_MODULE(TrackClusterRemover);
