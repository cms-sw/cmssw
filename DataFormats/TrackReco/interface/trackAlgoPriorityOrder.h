#ifndef DataFormats_TrackReco_trackAlgoPriorityOrder_h
#define DataFormats_TrackReco_trackAlgoPriorityOrder_h

#include "DataFormats/TrackReco/interface/TrackBase.h"

#include <array>

/**
 * The trackAlgoPriorityOrder maps an reco::TrackBase::TrackAlgorithm
 * enumerator to its priority in track list merging. The mapping is
 * needed because the order of the enumerators themselves does not, in
 * general, convey useful information.
 */
namespace impl {

  /**
   * This array defines the priority order in merging for the
   * algorithms. The order is ascending, i.e. the first algorithm has
   * the highest priority etc. The size of the array should be
   * reco::TrackBase:algoSize (checked below with static_assert), and
   * each reco::TrackBase::TrackAlgorithm enumerator should be in the
   * array exactly once (checked below in findIndex() function).
   */
  constexpr reco::TrackBase::TrackAlgorithm algoPriorityOrder[] ={
    reco::TrackBase::undefAlgorithm,
    reco::TrackBase::ctf,
    reco::TrackBase::rs,
    reco::TrackBase::cosmics,
    reco::TrackBase::initialStep,
    reco::TrackBase::detachedTripletStep,
    reco::TrackBase::lowPtTripletStep,
    reco::TrackBase::pixelPairStep,
    reco::TrackBase::mixedTripletStep,
    reco::TrackBase::pixelLessStep,
    reco::TrackBase::tobTecStep,
    reco::TrackBase::jetCoreRegionalStep,
    reco::TrackBase::conversionStep,
    reco::TrackBase::muonSeededStepInOut,
    reco::TrackBase::muonSeededStepOutIn,
    reco::TrackBase::outInEcalSeededConv,
    reco::TrackBase::inOutEcalSeededConv,
    reco::TrackBase::nuclInter,
    reco::TrackBase::standAloneMuon,
    reco::TrackBase::globalMuon,
    reco::TrackBase::cosmicStandAloneMuon,
    reco::TrackBase::cosmicGlobalMuon,
    reco::TrackBase::iter1LargeD0,
    reco::TrackBase::iter2LargeD0,
    reco::TrackBase::iter3LargeD0,
    reco::TrackBase::iter4LargeD0,
    reco::TrackBase::iter5LargeD0,
    reco::TrackBase::bTagGhostTracks,
    reco::TrackBase::beamhalo,
    reco::TrackBase::gsf,
    reco::TrackBase::hltPixel,
    reco::TrackBase::hltIter0,
    reco::TrackBase::hltIter1,
    reco::TrackBase::hltIter2,
    reco::TrackBase::hltIter3,
    reco::TrackBase::hltIter4,
    reco::TrackBase::hltIterX,
    reco::TrackBase::hiRegitMuInitialStep,
    reco::TrackBase::hiRegitMuPixelPairStep,
    reco::TrackBase::hiRegitMuMixedTripletStep,
    reco::TrackBase::hiRegitMuPixelLessStep,
    reco::TrackBase::hiRegitMuDetachedTripletStep,
    reco::TrackBase::hiRegitMuMuonSeededStepInOut,
    reco::TrackBase::hiRegitMuMuonSeededStepOutIn,
    reco::TrackBase::hiRegitMuLowPtTripletStep,
    reco::TrackBase::hiRegitMuTobTecStep
  };

  static_assert(reco::TrackBase::algoSize == sizeof(algoPriorityOrder)/sizeof(unsigned int), "Please update me too after adding new enumerators to reco::TrackBase::TrackAlgorithm");

  /**
   * Recursive implementation of searching the index of an algorithm in the algoPriorityOrder
   *
   * @param algo   Algorithm whose index is searched for
   * @param index  Current index
   *
   * @return Index of the algorithm; if not found results compile error
   */
  constexpr unsigned int findIndex(const reco::TrackBase::TrackAlgorithm algo, const unsigned int index) {
    return index < sizeof(algoPriorityOrder)/sizeof(unsigned int) ?
      (algo == algoPriorityOrder[index] ? index : findIndex(algo, index+1)) :
      throw "Index out of bounds, this means that some reco::TrackBase::TrackAlgorithm enumerator is missing from impl::algoPriorityOrder array.";
  }

  /**
   * Find the order priority for a track algorithm
   *
   * @param algo  algorithm whose index is searched for
   *
   * @return Index of the algorithm in impl::algoPriorityOrder array; if not found results compile error
   *
   * @see findIndex()
   */
  constexpr unsigned int priorityForAlgo(const reco::TrackBase::TrackAlgorithm algo) {
    return findIndex(algo, 0);
  }


  /**
   * Helper template to initialize std::array compile-time.
   *
   * Idea is that it "loops" over all reco::TrackBase::TrackAlgorithm
   * enumerators from end to beginning. In each "iteration", the order
   * priority is obtained from impl::algoPriorityOrder array, and the
   * priority is added to a parameter pack. When the beginning is
   * reached (termination condition is a partial specialization, see
   * below), the std::array is initialized from the parameter pack.
   * The "looping" is implemented as recursion.
   *
   * @tparam T  value_type of the std::array
   * @tparam N  Size of the std::array
   * @tparam I  Current index
   */
  template <typename T, size_t N, size_t I>
  struct MakeArray {
    template <typename ...Args>
    constexpr static
    std::array<T, N> value(Args&&... args) {
      return MakeArray<T, N, I-1>::value(priorityForAlgo(static_cast<reco::TrackBase::TrackAlgorithm>(I-1)), std::forward<Args>(args)...);
    }
  };

  /**
   * Partial specialization for the termination condition.
   */
  template <typename T, size_t N>
  struct MakeArray<T, N, 0> {
    template <typename ...Args>
    constexpr static
    std::array<T, N> value(Args&&... args) {
      return std::array<T, N>{{std::forward<Args>(args)...}};
    }
  };


  /**
   * Create compile-time an std::array mapping
   * reco::TrackBase::TrackAlgorithm enumerators to their order
   * priorities as defined in impl::algoPriorityOrder array.
   *
   * @tparam T  value_type of the std::array
   * @tparam N  Size of the std::array
   */
  template <typename T, size_t N>
  constexpr
  std::array<T, N> makeArray() {
    return MakeArray<T, N, N>::value();
  }

}

/**
 * Array mapping reco::TrackBase::TrackAlgorithm enumerators to their
 * order priorities in track list merging.
 */
constexpr std::array<unsigned int, reco::TrackBase::algoSize> trackAlgoPriorityOrder = impl::makeArray<unsigned int, reco::TrackBase::algoSize>();


#endif // DataFormats_TrackReco_trackAlgoPriorityOrder_h

