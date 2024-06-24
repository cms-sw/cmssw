#ifndef Event_cuh
#define Event_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"
#else
#include "Constants.h"
#include "Module.h"
#include "LST.h"
#endif

#include "Hit.h"
#include "ModuleMethods.h"
#include "Segment.h"
#include "Triplet.h"
#include "Kernels.h"
#include "Quintuplet.h"
#include "MiniDoublet.h"
#include "PixelTriplet.h"
#include "TrackCandidate.h"

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

namespace SDL {
  template <typename TAcc>
  class Event {};

  template <>
  class Event<SDL::Acc> {
  private:
    QueueAcc queue;
    Dev devAcc;
    DevHost devHost;
    bool addObjects;

    std::array<unsigned int, 6> n_hits_by_layer_barrel_;
    std::array<unsigned int, 5> n_hits_by_layer_endcap_;
    std::array<unsigned int, 6> n_minidoublets_by_layer_barrel_;
    std::array<unsigned int, 5> n_minidoublets_by_layer_endcap_;
    std::array<unsigned int, 6> n_segments_by_layer_barrel_;
    std::array<unsigned int, 5> n_segments_by_layer_endcap_;
    std::array<unsigned int, 6> n_triplets_by_layer_barrel_;
    std::array<unsigned int, 5> n_triplets_by_layer_endcap_;
    std::array<unsigned int, 6> n_trackCandidates_by_layer_barrel_;
    std::array<unsigned int, 5> n_trackCandidates_by_layer_endcap_;
    std::array<unsigned int, 6> n_quintuplets_by_layer_barrel_;
    std::array<unsigned int, 5> n_quintuplets_by_layer_endcap_;

    //Device stuff
    unsigned int nTotalSegments;
    struct objectRanges* rangesInGPU;
    struct objectRangesBuffer<Dev>* rangesBuffers;
    struct hits* hitsInGPU;
    struct hitsBuffer<Dev>* hitsBuffers;
    struct miniDoublets* mdsInGPU;
    struct miniDoubletsBuffer<Dev>* miniDoubletsBuffers;
    struct segments* segmentsInGPU;
    struct segmentsBuffer<Dev>* segmentsBuffers;
    struct triplets* tripletsInGPU;
    struct tripletsBuffer<Dev>* tripletsBuffers;
    struct quintuplets* quintupletsInGPU;
    struct quintupletsBuffer<Dev>* quintupletsBuffers;
    struct trackCandidates* trackCandidatesInGPU;
    struct trackCandidatesBuffer<Dev>* trackCandidatesBuffers;
    struct pixelTriplets* pixelTripletsInGPU;
    struct pixelTripletsBuffer<Dev>* pixelTripletsBuffers;
    struct pixelQuintuplets* pixelQuintupletsInGPU;
    struct pixelQuintupletsBuffer<Dev>* pixelQuintupletsBuffers;

    //CPU interface stuff
    objectRangesBuffer<alpaka::DevCpu>* rangesInCPU;
    hitsBuffer<alpaka::DevCpu>* hitsInCPU;
    miniDoubletsBuffer<alpaka::DevCpu>* mdsInCPU;
    segmentsBuffer<alpaka::DevCpu>* segmentsInCPU;
    tripletsBuffer<alpaka::DevCpu>* tripletsInCPU;
    trackCandidatesBuffer<alpaka::DevCpu>* trackCandidatesInCPU;
    modulesBuffer<alpaka::DevCpu>* modulesInCPU;
    quintupletsBuffer<alpaka::DevCpu>* quintupletsInCPU;
    pixelTripletsBuffer<alpaka::DevCpu>* pixelTripletsInCPU;
    pixelQuintupletsBuffer<alpaka::DevCpu>* pixelQuintupletsInCPU;

    void init(bool verbose);

    int* superbinCPU;
    int8_t* pixelTypeCPU;

    // Stuff that used to be global
    const uint16_t nModules_;
    const uint16_t nLowerModules_;
    const unsigned int nPixels_;
    const std::shared_ptr<const modulesBuffer<Dev>> modulesBuffers_;
    const std::shared_ptr<const pixelMap> pixelMapping_;
    const std::shared_ptr<const EndcapGeometry<Dev>> endcapGeometry_;

  public:
    // Constructor used for CMSSW integration. Uses an external queue.
    template <typename TQueue>
    Event(bool verbose, TQueue const& q, const LSTESDeviceData<Dev>* deviceESData)
        : queue(q),
          devAcc(alpaka::getDev(q)),
          devHost(cms::alpakatools::host()),
          nModules_(deviceESData->nModules),
          nLowerModules_(deviceESData->nLowerModules),
          nPixels_(deviceESData->nPixels),
          modulesBuffers_(deviceESData->modulesBuffers),
          pixelMapping_(deviceESData->pixelMapping),
          endcapGeometry_(deviceESData->endcapGeometry) {
      init(verbose);
    }
    void resetEvent();

    void addHitToEvent(
        std::vector<float> x,
        std::vector<float> y,
        std::vector<float> z,
        std::vector<unsigned int> detId,
        std::vector<unsigned int> idxInNtuple);  //call the appropriate hit function, then increment the counter here
    void addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,
                                std::vector<unsigned int> hitIndices1,
                                std::vector<unsigned int> hitIndices2,
                                std::vector<unsigned int> hitIndices3,
                                std::vector<float> dPhiChange,
                                std::vector<float> ptIn,
                                std::vector<float> ptErr,
                                std::vector<float> px,
                                std::vector<float> py,
                                std::vector<float> pz,
                                std::vector<float> eta,
                                std::vector<float> etaErr,
                                std::vector<float> phi,
                                std::vector<int> charge,
                                std::vector<unsigned int> seedIdx,
                                std::vector<int> superbin,
                                std::vector<int8_t> pixelType,
                                std::vector<char> isQuad);

    // functions that map the objects to the appropriate modules
    void addMiniDoubletsToEventExplicit();
    void addSegmentsToEventExplicit();
    void addTripletsToEventExplicit();
    void addQuintupletsToEventExplicit();
    void resetObjectsInModule();

    void createMiniDoublets();
    void createSegmentsWithModuleMap();
    void createTriplets();
    void createPixelTracklets();
    void createPixelTrackletsWithMap();
    void createTrackCandidates();
    void createExtendedTracks();
    void createQuintuplets();
    void createPixelTriplets();
    void createPixelQuintuplets();
    void pixelLineSegmentCleaning();

    unsigned int getNumberOfHits();
    unsigned int getNumberOfHitsByLayer(unsigned int layer);
    unsigned int getNumberOfHitsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfHitsByLayerEndcap(unsigned int layer);

    unsigned int getNumberOfMiniDoublets();
    unsigned int getNumberOfMiniDoubletsByLayer(unsigned int layer);
    unsigned int getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer);

    unsigned int getNumberOfSegments();
    unsigned int getNumberOfSegmentsByLayer(unsigned int layer);
    unsigned int getNumberOfSegmentsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfSegmentsByLayerEndcap(unsigned int layer);

    unsigned int getNumberOfTriplets();
    unsigned int getNumberOfTripletsByLayer(unsigned int layer);
    unsigned int getNumberOfTripletsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfTripletsByLayerEndcap(unsigned int layer);

    int getNumberOfTrackCandidates();
    int getNumberOfPixelTrackCandidates();
    int getNumberOfPT5TrackCandidates();
    int getNumberOfPT3TrackCandidates();
    int getNumberOfT5TrackCandidates();
    int getNumberOfPLSTrackCandidates();

    unsigned int getNumberOfQuintuplets();
    unsigned int getNumberOfQuintupletsByLayer(unsigned int layer);
    unsigned int getNumberOfQuintupletsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfQuintupletsByLayerEndcap(unsigned int layer);

    int getNumberOfPixelTriplets();
    int getNumberOfPixelQuintuplets();

    objectRangesBuffer<alpaka::DevCpu>* getRanges();
    hitsBuffer<alpaka::DevCpu>* getHits();
    hitsBuffer<alpaka::DevCpu>* getHitsInCMSSW();
    miniDoubletsBuffer<alpaka::DevCpu>* getMiniDoublets();
    segmentsBuffer<alpaka::DevCpu>* getSegments();
    tripletsBuffer<alpaka::DevCpu>* getTriplets();
    quintupletsBuffer<alpaka::DevCpu>* getQuintuplets();
    trackCandidatesBuffer<alpaka::DevCpu>* getTrackCandidates();
    trackCandidatesBuffer<alpaka::DevCpu>* getTrackCandidatesInCMSSW();
    pixelTripletsBuffer<alpaka::DevCpu>* getPixelTriplets();
    pixelQuintupletsBuffer<alpaka::DevCpu>* getPixelQuintuplets();
    modulesBuffer<alpaka::DevCpu>* getModules(bool isFull = false);
  };

}  // namespace SDL
#endif
