#ifndef RecoLocalTracker_SiPixelClusterizer_alpaka_PixelClustering_h
#define RecoLocalTracker_SiPixelClusterizer_alpaka_PixelClustering_h

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "FWCore/Utilities/interface/DeviceGlobal.h"
#include "FWCore/Utilities/interface/HostDeviceConstant.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageSoA.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageDevice.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelClustering {

#ifdef GPU_DEBUG
  DEVICE_GLOBAL uint32_t gMaxHit = 0;
#endif

  namespace pixelStatus {
    // Phase-1 pixel modules
    constexpr uint32_t pixelSizeX = pixelTopology::Phase1::numRowsInModule;
    constexpr uint32_t pixelSizeY = pixelTopology::Phase1::numColsInModule;
    constexpr int32_t empVal = std::numeric_limits<int32_t>::max() - 2; //TODO: Move this to DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h
    constexpr int32_t fakeVal = 9999;//std::numeric_limits<int32_t>::max() - 3; //TODO: Move this to DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h

    // Use 0x00, 0x01, 0x03 so each can be OR'ed on top of the previous ones
    enum Status : uint32_t { kEmpty = 0x00, kFound = 0x01, kDuplicate = 0x03 };

    constexpr uint32_t bits = 2;
    constexpr uint32_t mask = (0x01 << bits) - 1;
    constexpr uint32_t valuesPerWord = sizeof(uint32_t) * 8 / bits;
    constexpr uint32_t size = pixelSizeX * pixelSizeY / valuesPerWord;

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr uint32_t getIndex(uint16_t x, uint16_t y) {
      return (pixelSizeX * y + x) / valuesPerWord;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr uint32_t getShift(uint16_t x, uint16_t y) {
      return (x % valuesPerWord) * 2;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr Status getStatus(uint32_t const* __restrict__ status,
                                                              uint16_t x,
                                                              uint16_t y) {
      uint32_t index = getIndex(x, y);
      uint32_t shift = getShift(x, y);
      return Status{(status[index] >> shift) & mask};
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr bool isDuplicate(uint32_t const* __restrict__ status,
                                                              uint16_t x,
                                                              uint16_t y) {
      return getStatus(status, x, y) == kDuplicate;
    }

    /* FIXME
       * In the more general case (e.g. a multithreaded CPU backend) there is a potential race condition
       * between the read of status[index] at line NNN and the atomicCas at line NNN.
       * We should investigate:
       *   - if `status` should be read through a `volatile` pointer (CUDA/ROCm)
       *   - if `status` should be read with an atomic load (CPU)
       */
    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr void promote(Acc1D const& acc,
                                                          uint32_t* __restrict__ status,
                                                          const uint16_t x,
                                                          const uint16_t y) {
      uint32_t index = getIndex(x, y);
      uint32_t shift = getShift(x, y);
      uint32_t old_word = status[index];
      uint32_t expected = old_word;
      do {
        expected = old_word;
        Status old_status{(old_word >> shift) & mask};
        if (kDuplicate == old_status) {
          // nothing to do
          return;
        }
        Status new_status = (kEmpty == old_status) ? kFound : kDuplicate;
        uint32_t new_word = old_word | (static_cast<uint32_t>(new_status) << shift);
        old_word = alpaka::atomicCas(acc, &status[index], expected, new_word, alpaka::hierarchy::Blocks{});
      } while (expected != old_word);
    }

  }  // namespace pixelStatus

  template <typename TrackerTraits>
  struct CountModules {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAView digi_view,
                                  SiPixelClustersSoAView clus_view,
                                  const unsigned int numElements) const {
      // Make sure the atomicInc below does not overflow
      static_assert(TrackerTraits::numberOfModules < ::pixelClustering::maxNumModules);

#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("Starting to count modules to set module starts:");
      }
#endif
      for (int32_t i : cms::alpakatools::uniform_elements(acc, numElements)) {
        digi_view[i].clus() = i;
        if (::pixelClustering::invalidModuleId == digi_view[i].moduleId())
          continue;

        int32_t j = i - 1;
        while (j >= 0 and digi_view[j].moduleId() == ::pixelClustering::invalidModuleId)
          --j;
        if (j < 0 or digi_view[j].moduleId() != digi_view[i].moduleId()) {
          // Found a module boundary: count the number of modules in  clus_view[0].moduleStart()
          auto loc = alpaka::atomicInc(acc,
                                       &clus_view[0].moduleStart(),
                                       static_cast<uint32_t>(::pixelClustering::maxNumModules),
                                       alpaka::hierarchy::Blocks{});
          ALPAKA_ASSERT_ACC(loc < TrackerTraits::numberOfModules);
#ifdef GPU_DEBUG
          printf("> New module (no. %d) found at digi %d \n", loc, i);
#endif
          clus_view[loc + 1].moduleStart() = i;
        }
      }
    }
  };

/**
  template <typename TrackerTraits>
  struct SparseToDense {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAView digi_view,
                                  SiPixelClustersSoAView clus_view,
                                  const unsigned int numElements) const {
      // Make sure the atomicInc below does not overflow
      static_assert(TrackerTraits::numberOfModules < ::pixelClustering::maxNumModules);
      auto& lastPixel = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

      const uint32_t lastModule = clus_view[0].moduleStart();
      for (uint32_t module : cms::alpakatools::independent_groups(acc, lastModule)) {
        auto firstPixel = clus_view[1 + module].moduleStart();
        uint32_t thisModuleId = digi_view[firstPixel].moduleId();
        ALPAKA_ASSERT_ACC(thisModuleId < TrackerTraits::numberOfModules);
        lastPixel = numElements;
        alpaka::syncBlockThreads(acc);

        // skip threads not associated to an existing pixel
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, numElements)) {
          auto id = digi_view[i].moduleId();
          // skip invalid pixels
          if (id == ::pixelClustering::invalidModuleId)
            continue;
          // find the first pixel in a different module
          if (id != thisModuleId) {
            alpaka::atomicMin(acc, &lastPixel, i, alpaka::hierarchy::Threads{});
            break;
          }
        }
	const unsigned int lp = *lastPixel;
        if (cms::alpakatools::once_per_block(acc)){
		printf("FirstPixel %u",firstPixel);
		printf("LastPixel %u",lp);
		for(auto i = firstPixel ; i<lastPixel ; i++)
		{
			printf("Pixel (%u,%u) in module %u",digi_view[i].xx(),digi_view[i].yy(),thisModuleId);
		}
	}

}}
};
**/


  template <typename TrackerTraits>
  struct FindClus {
    // assume that we can cover the whole module with up to 16 blockDimension-wide iterations
    static constexpr uint32_t maxIterGPU = 16;

    // this must be larger than maxPixInModule / maxIterGPU, and should be a multiple of the warp size
    static constexpr uint32_t maxElementsPerBlock =
        cms::alpakatools::round_up_by(TrackerTraits::maxPixInModule / maxIterGPU, 128);

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAView digi_view,
		    		  SiPixelImageSoAView images,
                                  SiPixelClustersSoAView clus_view,
                                  const unsigned int numElements) const {
      static_assert(TrackerTraits::numberOfModules < ::pixelClustering::maxNumModules);

      auto& lastPixel = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

      const uint32_t lastModule = clus_view[0].moduleStart();
      for (uint32_t module : cms::alpakatools::independent_groups(acc, lastModule)) {
        auto firstPixel = clus_view[1 + module].moduleStart();
        uint32_t thisModuleId = digi_view[firstPixel].moduleId();
        ALPAKA_ASSERT_ACC(thisModuleId < TrackerTraits::numberOfModules);

#ifdef GPU_DEBUG
        if (thisModuleId % 100 == 1)
          if (cms::alpakatools::once_per_block(acc))
            printf("start clusterizer for module %4d in block %4d\n",
                   thisModuleId,
                   alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
#endif

        // find the index of the first pixel not belonging to this module (or invalid)
        lastPixel = numElements;
        alpaka::syncBlockThreads(acc);

        // skip threads not associated to an existing pixel
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, numElements)) {
          auto id = digi_view[i].moduleId();
          // skip invalid pixels
          if (id == ::pixelClustering::invalidModuleId)
            continue;
          // find the first pixel in a different module
          if (id != thisModuleId) {
            alpaka::atomicMin(acc, &lastPixel, i, alpaka::hierarchy::Threads{});
            break;
          }
        }


        ALPAKA_ASSERT_ACC((lastPixel == numElements) or
                          ((lastPixel < numElements) and (digi_view[lastPixel].moduleId() != thisModuleId)));
        // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
        if (cms::alpakatools::once_per_block(acc)) {
          if (lastPixel - firstPixel > TrackerTraits::maxPixInModule) {
            printf("too many pixels in module %u: %u > %u\n",
                   thisModuleId,
                   lastPixel - firstPixel,
                   TrackerTraits::maxPixInModule);
            lastPixel = TrackerTraits::maxPixInModule + firstPixel;
          }
        }
        alpaka::syncBlockThreads(acc);
        ALPAKA_ASSERT_ACC(lastPixel - firstPixel <= TrackerTraits::maxPixInModule);

	uint32_t pixsize = pixelStatus::pixelSizeX * pixelStatus::pixelSizeY;
        for (uint32_t j : cms::alpakatools::independent_group_elements(acc, pixsize))
	{
		uint16_t row = j/pixelStatus::pixelSizeY;
		uint16_t col = j%pixelStatus::pixelSizeY;
		images[module].clus()[col][row] = pixelStatus::empVal;
	}
        alpaka::syncBlockThreads(acc);

        // remove duplicate pixels
        constexpr bool isPhase2 = std::is_base_of<pixelTopology::Phase2, TrackerTraits>::value;
	for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel))
	{
		 if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
		   continue;
		 int fpX = digi_view[i].xx();
		 int fpY = digi_view[i].yy();
		if constexpr (not isPhase2)
		{
			 int32_t kEmp = pixelStatus::empVal;
			 int32_t old_value = alpaka::atomicCas(acc, &images[module].clus()[fpY][fpX],kEmp, digi_view[i].clus());
			 if(old_value != pixelStatus::empVal)
			 {
				digi_view[i].moduleId() = ::pixelClustering::invalidModuleId;
				digi_view[i].rawIdArr() = 0;
			 }

		 }
		 else 
		 {
			 images[module].clus()[fpY][fpX] = digi_view[i].clus();
		 }

	}
	alpaka::syncBlockThreads(acc);

#if 0
	if(module == 100)
	{
		for (uint32_t j : cms::alpakatools::independent_group_elements(acc, pixsize))
		{
			uint16_t row = j/pixelStatus::pixelSizeY;
			uint16_t col = j%pixelStatus::pixelSizeY;
			if(row<80 && col<52)
			{
				printf("(%u,%u):%u/",row,col,images[module].clus()[row][col]);
			}
		}
		alpaka::syncBlockThreads(acc);
		printf("\n");
		uint32_t pixsize = pixelStatus::pixelSizeX * pixelStatus::pixelSizeY;
		uint32_t kernal[3][3] = { {1,1,1},{1,1,1},{1,1,1} };
		for (uint32_t j : cms::alpakatools::independent_group_elements(acc, pixsize))
		{
			uint16_t row = j/pixelStatus::pixelSizeY;
			uint16_t col = j%pixelStatus::pixelSizeY;
			auto thisPix = images[module].clus()[row][col];
			for( int i = 0 ; i<3 && images[module].clus()[row][col] == pixelStatus::empVal ; i++)
			{
				for( int j = 0 ; j<3; j++)
				{
					if(images[module].clus()[row+i-1][col+j-1] != pixelStatus::empVal && images[module].clus()[row+i-1][col+j-1] != pixelStatus::fakeVal && kernal[i][j])
					{
						images[module].clus()[row][col] = pixelStatus::fakeVal;
						break;
					}
				}
			}

		}
		alpaka::syncBlockThreads(acc);
		for (uint32_t j : cms::alpakatools::independent_group_elements(acc, pixsize))
		{
			uint16_t row = j/pixelStatus::pixelSizeY;
			uint16_t col = j%pixelStatus::pixelSizeY;
			if(row<80 && col<52)
			{
				printf("(%u,%u):%u/",row,col,images[module].clus()[row][col]);
			}
		}
		printf("\n");
		alpaka::syncBlockThreads(acc);
		for (uint32_t j : cms::alpakatools::independent_group_elements(acc, pixsize))
		{
			uint16_t row = j/pixelStatus::pixelSizeY;
			uint16_t col = j%pixelStatus::pixelSizeY;
			auto thisPix = images[module].clus()[row][col];
			for( int i = 0 ; i<3 && images[module].clus()[row][col] != pixelStatus::empVal ; i++)
			{
				for( int j = 0 ; j<3; j++)
				{
					if(images[module].clus()[row+i-1][col+j-1] == pixelStatus::empVal && kernal[i][j])
					{
						images[module].clus()[row][col] = pixelStatus::empVal;
						break;
					}
				}
			}

		}
		alpaka::syncBlockThreads(acc);
		for (uint32_t j : cms::alpakatools::independent_group_elements(acc, pixsize))
		{
			uint16_t row = j/pixelStatus::pixelSizeY;
			uint16_t col = j%pixelStatus::pixelSizeY;
			if(row<80 && col<52)
			{
				printf("(%u,%u):%u/",row,col,images[module].clus()[row][col]);
			}
		}
	}
#endif


        // for each pixel, look at all the pixels until the end of the module;
        // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
        // after the loop, all the pixel in each cluster should have the id equeal to the lowest
        // pixel in the cluster ( clus[i] == i ).
        bool more = true;
        while (alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, more)) {
          more = false;
          for (uint32_t j : cms::alpakatools::independent_group_elements(acc, pixsize)){

		uint16_t row = j/pixelStatus::pixelSizeY;
		uint16_t col = j%pixelStatus::pixelSizeY;
		if( images[module].clus()[col][row] == pixelStatus::empVal  || row >= pixelStatus::pixelSizeX - 1 || col >= pixelStatus::pixelSizeY - 1 || row<=1 || col<= 1)
		{
			//more=true;
			continue;
		}
		int32_t tmp = pixelStatus::empVal ;

		for (int kk = -1; kk < 2; kk++) {
		    for (int jj = -1; jj < 2; jj++) {
			    int32_t clus = images[module].clus()[col+kk][row+jj];
			    tmp= alpaka::math::min(acc,tmp,clus);
		    }
		}
		if(images[module].clus()[col][row] != tmp)
		{
			images[module].clus()[col][row] = tmp;
			more=true;
		}
	}
		
          alpaka::syncBlockThreads(acc);
#if 0
          for (uint32_t j : cms::alpakatools::independent_group_elements(acc, hist.size())) {
            auto p = hist.begin() + j;
            auto i = *p + firstPixel;
            auto m = digi_view[i].clus();
            while (m != digi_view[m].clus())
              m = digi_view[m].clus();
            digi_view[i].clus() = m;
          }
#endif
        }  // end while

	alpaka::syncBlockThreads(acc);
	for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel))
	{
		 int fpX = digi_view[i].xx();
		 int fpY = digi_view[i].yy();
		 if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
		   continue;
		 digi_view[i].clus() = images[module].clus()[fpY][fpX];

		 //if(images[module].clus()[fpY][fpX]==digi_view[i].clus())
		// {
		//	 digi_view[i].clus() = images[module].clus()[fpY][fpX];
			 //printf("COOL  %u , %u", digi_view[i].clus(),images[module].clus()[fpY][fpX]);
		//	 }
		 //else
		// {
		//	 printf("SHITTTT %u , %u", digi_view[i].clus(),images[module].clus()[fpY][fpX]);
		// }
		 //printf("CLUSID: %d\n",images[module].clus()[fpY][fpX]);
	}

	  alpaka::syncBlockThreads(acc);
        /*
            // check that all threads in the block have executed the same number of iterations
            auto& n0 = alpaka::declareSharedVar<int, __COUNTER__>(acc);
            if (cms::alpakatools::once_per_block(acc))
              n0 = nloops;
            alpaka::syncBlockThreads(acc);
            ALPAKA_ASSERT_ACC(alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, nloops == n0));
            if (thisModuleId % 100 == 1)
              if (cms::alpakatools::once_per_block(acc))
                printf("# loops %d\n", nloops);
          */
        auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        foundClusters = 0;
        alpaka::syncBlockThreads(acc);

        // find the number of different clusters, identified by a pixels with clus[i] == i;
        // mark these pixels with a negative id.
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid pixels
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
            continue;
          if (digi_view[i].clus() == static_cast<int>(i)) {
            auto old = alpaka::atomicInc(acc, &foundClusters, 0xffffffff, alpaka::hierarchy::Threads{});
            digi_view[i].clus() = -(old + 1);
          }
        }
        alpaka::syncBlockThreads(acc);

        // propagate the negative id to all the pixels in the cluster.
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid pixels
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
            continue;
          if (digi_view[i].clus() >= 0) {
            // mark each pixel in a cluster with the same id as the first one
            digi_view[i].clus() = digi_view[digi_view[i].clus()].clus();
          }
        }
        alpaka::syncBlockThreads(acc);

        // adjust the cluster id to be a positive value starting from 0
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId) {
            // mark invalid pixels with an invalid cluster index
            digi_view[i].clus() = ::pixelClustering::invalidClusterId;
          } else {
            digi_view[i].clus() = -digi_view[i].clus() - 1;
          }
        }
        alpaka::syncBlockThreads(acc);
        if (cms::alpakatools::once_per_block(acc)) {
          clus_view[thisModuleId].clusInModule() = foundClusters;
          clus_view[module].moduleId() = thisModuleId;
        }
      }  // module loop
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelClustering

#endif  // plugin_SiPixelClusterizer_alpaka_PixelClustering.h
