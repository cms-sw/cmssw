#ifndef RecoTracker_LSTCore_src_alpaka_LSTEvent_h
#define RecoTracker_LSTCore_src_alpaka_LSTEvent_h

#include <optional>

#include "RecoTracker/LSTCore/interface/HitsHostCollection.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsHostCollection.h"
#include "RecoTracker/LSTCore/interface/PixelQuintupletsHostCollection.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsHostCollection.h"
#include "RecoTracker/LSTCore/interface/QuintupletsHostCollection.h"
#include "RecoTracker/LSTCore/interface/SegmentsHostCollection.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesHostCollection.h"
#include "RecoTracker/LSTCore/interface/TripletsHostCollection.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesHostCollection.h"
#include "RecoTracker/LSTCore/interface/ModulesHostCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"
#include "RecoTracker/LSTCore/interface/alpaka/MiniDoubletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/PixelQuintupletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/PixelTripletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/QuintupletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/SegmentsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TripletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/ModulesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/ObjectRangesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/EndcapGeometryDevDeviceCollection.h"

#include "Hit.h"
#include "Kernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  class LSTEvent {
  private:
    Queue& queue_;

    std::array<unsigned int, 6> n_minidoublets_by_layer_barrel_{};
    std::array<unsigned int, 5> n_minidoublets_by_layer_endcap_{};
    std::array<unsigned int, 6> n_segments_by_layer_barrel_{};
    std::array<unsigned int, 5> n_segments_by_layer_endcap_{};
    std::array<unsigned int, 6> n_triplets_by_layer_barrel_{};
    std::array<unsigned int, 5> n_triplets_by_layer_endcap_{};
    std::array<unsigned int, 6> n_quintuplets_by_layer_barrel_{};
    std::array<unsigned int, 5> n_quintuplets_by_layer_endcap_{};
    unsigned int nTotalSegments_;

    //Device stuff
    std::optional<ObjectRangesDeviceCollection> rangesDC_;
    std::optional<HitsDeviceCollection> hitsDC_;
    std::optional<MiniDoubletsDeviceCollection> miniDoubletsDC_;
    std::optional<SegmentsDeviceCollection> segmentsDC_;
    std::optional<TripletsDeviceCollection> tripletsDC_;
    std::optional<QuintupletsDeviceCollection> quintupletsDC_;
    std::optional<TrackCandidatesDeviceCollection> trackCandidatesDC_;
    std::optional<PixelTripletsDeviceCollection> pixelTripletsDC_;
    std::optional<PixelQuintupletsDeviceCollection> pixelQuintupletsDC_;

    //CPU interface stuff
    std::optional<ObjectRangesHostCollection> rangesHC_;
    std::optional<HitsHostCollection> hitsHC_;
    std::optional<MiniDoubletsHostCollection> miniDoubletsHC_;
    std::optional<SegmentsHostCollection> segmentsHC_;
    std::optional<TripletsHostCollection> tripletsHC_;
    std::optional<TrackCandidatesHostCollection> trackCandidatesHC_;
    std::optional<ModulesHostCollection> modulesHC_;
    std::optional<QuintupletsHostCollection> quintupletsHC_;
    std::optional<PixelTripletsHostCollection> pixelTripletsHC_;
    std::optional<PixelQuintupletsHostCollection> pixelQuintupletsHC_;

    const uint16_t nModules_;
    const uint16_t nLowerModules_;
    const unsigned int nPixels_;
    const unsigned int nEndCapMap_;
    ModulesDeviceCollection const& modules_;
    PixelMap const& pixelMapping_;
    EndcapGeometryDevDeviceCollection const& endcapGeometry_;
    bool addObjects_;

  public:
    // Constructor used for CMSSW integration. Uses an external queue.
    LSTEvent(bool verbose, Queue& q, const LSTESData<Device>* deviceESData)
        : queue_(q),
          nModules_(deviceESData->nModules),
          nLowerModules_(deviceESData->nLowerModules),
          nPixels_(deviceESData->nPixels),
          nEndCapMap_(deviceESData->nEndCapMap),
          modules_(*deviceESData->modules),
          pixelMapping_(*deviceESData->pixelMapping),
          endcapGeometry_(*deviceESData->endcapGeometry),
          addObjects_(verbose) {}
    void initSync();        // synchronizes, for standalone usage
    void resetEventSync();  // synchronizes, for standalone usage
    void wait() const { alpaka::wait(queue_); }

    // Calls the appropriate hit function, then increments the counter
    void addHitToEvent(std::vector<float> const& x,
                       std::vector<float> const& y,
                       std::vector<float> const& z,
                       std::vector<unsigned int> const& detId,
                       std::vector<unsigned int> const& idxInNtuple);
    void addPixelSegmentToEvent(std::vector<unsigned int> const& hitIndices0,
                                std::vector<unsigned int> const& hitIndices1,
                                std::vector<unsigned int> const& hitIndices2,
                                std::vector<unsigned int> const& hitIndices3,
                                std::vector<float> const& dPhiChange,
                                std::vector<float> const& ptIn,
                                std::vector<float> const& ptErr,
                                std::vector<float> const& px,
                                std::vector<float> const& py,
                                std::vector<float> const& pz,
                                std::vector<float> const& eta,
                                std::vector<float> const& etaErr,
                                std::vector<float> const& phi,
                                std::vector<int> const& charge,
                                std::vector<unsigned int> const& seedIdx,
                                std::vector<int> const& superbin,
                                std::vector<PixelType> const& pixelType,
                                std::vector<char> const& isQuad);

    void createMiniDoublets();
    void createSegmentsWithModuleMap();
    void createTriplets();
    void createTrackCandidates(bool no_pls_dupclean, bool tc_pls_triplets);
    void createPixelTriplets();
    void createQuintuplets();
    void pixelLineSegmentCleaning(bool no_pls_dupclean);
    void createPixelQuintuplets();

    // functions that map the objects to the appropriate modules
    void addMiniDoubletsToEventExplicit();
    void addSegmentsToEventExplicit();
    void addQuintupletsToEventExplicit();
    void addTripletsToEventExplicit();
    void resetObjectsInModule();

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

    int getNumberOfPixelTriplets();
    int getNumberOfPixelQuintuplets();

    unsigned int getNumberOfQuintuplets();
    unsigned int getNumberOfQuintupletsByLayer(unsigned int layer);
    unsigned int getNumberOfQuintupletsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfQuintupletsByLayerEndcap(unsigned int layer);

    int getNumberOfTrackCandidates();
    int getNumberOfPT5TrackCandidates();
    int getNumberOfPT3TrackCandidates();
    int getNumberOfPLSTrackCandidates();
    int getNumberOfPixelTrackCandidates();
    int getNumberOfT5TrackCandidates();

    // sync adds alpaka::wait at the end of filling a buffer during lazy fill
    // (has no effect on repeated calls)
    // set to false may allow faster operation with concurrent calls of get*
    // HANDLE WITH CARE
    template <typename TSoA, typename TDev = Device>
    typename TSoA::ConstView getHits(bool inCMSSW = false, bool sync = true);
    template <typename TDev = Device>
    ObjectRangesConst getRanges(bool sync = true);
    template <typename TSoA, typename TDev = Device>
    typename TSoA::ConstView getMiniDoublets(bool sync = true);
    template <typename TSoA, typename TDev = Device>
    typename TSoA::ConstView getSegments(bool sync = true);
    template <typename TSoA, typename TDev = Device>
    typename TSoA::ConstView getTriplets(bool sync = true);
    template <typename TSoA, typename TDev = Device>
    typename TSoA::ConstView getQuintuplets(bool sync = true);
    template <typename TDev = Device>
    PixelTripletsConst getPixelTriplets(bool sync = true);
    template <typename TDev = Device>
    PixelQuintupletsConst getPixelQuintuplets(bool sync = true);
    const TrackCandidatesConst& getTrackCandidates(bool inCMSSW = false, bool sync = true);
    template <typename TSoA, typename TDev = Device>
    typename TSoA::ConstView getModules(bool sync = true);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
