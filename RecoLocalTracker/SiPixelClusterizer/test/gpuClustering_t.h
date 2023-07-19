#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#endif  // __CUDACC__

#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusterChargeCut.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelClusterThresholds.h"
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

int main(void) {
#ifdef __CUDACC__
  cms::cudatest::requireDevices();
#endif  // __CUDACC__

  using namespace gpuClustering;
  using pixelTopology::Phase1;

  constexpr int numElements = 256 * maxNumModules;
  const SiPixelClusterThresholds clusterThresholds(
      clusterThresholdLayerOne, clusterThresholdOtherLayers, 0.f, 0.f, 0.f, 0.f);

  // these in reality are already on GPU
  auto h_raw = std::make_unique<uint32_t[]>(numElements);
  auto h_id = std::make_unique<uint16_t[]>(numElements);
  auto h_x = std::make_unique<uint16_t[]>(numElements);
  auto h_y = std::make_unique<uint16_t[]>(numElements);
  auto h_adc = std::make_unique<uint16_t[]>(numElements);
  auto h_clus = std::make_unique<int[]>(numElements);

#ifdef __CUDACC__
  auto d_raw = cms::cuda::make_device_unique<uint32_t[]>(numElements, nullptr);
  auto d_id = cms::cuda::make_device_unique<uint16_t[]>(numElements, nullptr);
  auto d_x = cms::cuda::make_device_unique<uint16_t[]>(numElements, nullptr);
  auto d_y = cms::cuda::make_device_unique<uint16_t[]>(numElements, nullptr);
  auto d_adc = cms::cuda::make_device_unique<uint16_t[]>(numElements, nullptr);
  auto d_clus = cms::cuda::make_device_unique<int[]>(numElements, nullptr);
  auto d_moduleStart = cms::cuda::make_device_unique<uint32_t[]>(maxNumModules + 1, nullptr);
  auto d_clusInModule = cms::cuda::make_device_unique<uint32_t[]>(maxNumModules, nullptr);
  auto d_moduleId = cms::cuda::make_device_unique<uint32_t[]>(maxNumModules, nullptr);
#else   // __CUDACC__
  auto h_moduleStart = std::make_unique<uint32_t[]>(maxNumModules + 1);
  auto h_clusInModule = std::make_unique<uint32_t[]>(maxNumModules);
  auto h_moduleId = std::make_unique<uint32_t[]>(maxNumModules);
#endif  // __CUDACC__

  // later random number
  int n = 0;
  int ncl = 0;
  int y[10] = {5, 7, 9, 1, 3, 0, 4, 8, 2, 6};

  auto generateClusters = [&](int kn) {
    auto addBigNoise = 1 == kn % 2;
    if (addBigNoise) {
      constexpr int MaxPixels = 1000;
      int id = 666;
      for (int x = 0; x < 140; x += 3) {
        for (int yy = 0; yy < 400; yy += 3) {
          h_id[n] = id;
          h_x[n] = x;
          h_y[n] = yy;
          h_adc[n] = 1000;
          ++n;
          ++ncl;
          if (MaxPixels <= ncl)
            break;
        }
        if (MaxPixels <= ncl)
          break;
      }
    }

    {
      // isolated
      int id = 42;
      int x = 10;
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = x;
      h_adc[n] = kn == 0 ? 100 : 5000;
      ++n;

      // first column
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = 0;
      h_adc[n] = 5000;
      ++n;
      // first columns
      ++ncl;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 2;
      h_adc[n] = 5000;
      ++n;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 1;
      h_adc[n] = 5000;
      ++n;

      // last column
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = 415;
      h_adc[n] = 5000;
      ++n;
      // last columns
      ++ncl;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 415;
      h_adc[n] = 2500;
      ++n;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 414;
      h_adc[n] = 2500;
      ++n;

      // diagonal
      ++ncl;
      for (int x = 20; x < 25; ++x) {
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = x;
        h_adc[n] = 1000;
        ++n;
      }
      ++ncl;
      // reversed
      for (int x = 45; x > 40; --x) {
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = x;
        h_adc[n] = 1000;
        ++n;
      }
      ++ncl;
      h_id[n++] = invalidModuleId;  // error
      // messy
      int xx[5] = {21, 25, 23, 24, 22};
      for (int k = 0; k < 5; ++k) {
        h_id[n] = id;
        h_x[n] = xx[k];
        h_y[n] = 20 + xx[k];
        h_adc[n] = 1000;
        ++n;
      }
      // holes
      ++ncl;
      for (int k = 0; k < 5; ++k) {
        h_id[n] = id;
        h_x[n] = xx[k];
        h_y[n] = 100;
        h_adc[n] = kn == 2 ? 100 : 1000;
        ++n;
        if (xx[k] % 2 == 0) {
          h_id[n] = id;
          h_x[n] = xx[k];
          h_y[n] = 101;
          h_adc[n] = 1000;
          ++n;
        }
      }
    }
    {
      // id == 0 (make sure it works!
      int id = 0;
      int x = 10;
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = x;
      h_adc[n] = 5000;
      ++n;
    }
    // all odd id
    for (int id = 11; id <= 1800; id += 2) {
      if ((id / 20) % 2)
        h_id[n++] = invalidModuleId;  // error
      for (int x = 0; x < 40; x += 4) {
        ++ncl;
        if ((id / 10) % 2) {
          for (int k = 0; k < 10; ++k) {
            h_id[n] = id;
            h_x[n] = x;
            h_y[n] = x + y[k];
            h_adc[n] = 100;
            ++n;
            h_id[n] = id;
            h_x[n] = x + 1;
            h_y[n] = x + y[k] + 2;
            h_adc[n] = 1000;
            ++n;
          }
        } else {
          for (int k = 0; k < 10; ++k) {
            h_id[n] = id;
            h_x[n] = x;
            h_y[n] = x + y[9 - k];
            h_adc[n] = kn == 2 ? 10 : 1000;
            ++n;
            if (y[k] == 3)
              continue;  // hole
            if (id == 51) {
              h_id[n++] = invalidModuleId;
              h_id[n++] = invalidModuleId;
            }  // error
            h_id[n] = id;
            h_x[n] = x + 1;
            h_y[n] = x + y[k] + 2;
            h_adc[n] = kn == 2 ? 10 : 1000;
            ++n;
          }
        }
      }
    }
  };  // end lambda
  for (auto kkk = 0; kkk < 5; ++kkk) {
    n = 0;
    ncl = 0;
    generateClusters(kkk);

    std::cout << "created " << n << " digis in " << ncl << " clusters" << std::endl;
    assert(n <= numElements);

    uint32_t nModules = 0;
#ifdef __CUDACC__
    size_t size32 = n * sizeof(unsigned int);
    size_t size16 = n * sizeof(unsigned short);
    // size_t size8 = n * sizeof(uint8_t);

    cudaCheck(cudaMemcpy(d_moduleStart.get(), &nModules, sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_id.get(), h_id.get(), size16, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_x.get(), h_x.get(), size16, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_y.get(), h_y.get(), size16, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_adc.get(), h_adc.get(), size16, cudaMemcpyHostToDevice));

    // Launch CUDA Kernels
    int threadsPerBlock = (kkk == 5) ? 512 : ((kkk == 3) ? 128 : 256);
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA countModules kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock
              << " threads\n";

    cms::cuda::launch(
        countModules<Phase1>, {blocksPerGrid, threadsPerBlock}, d_id.get(), d_moduleStart.get(), d_clus.get(), n);

    blocksPerGrid = maxNumModules;  //nModules;

    std::cout << "CUDA findModules kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock
              << " threads\n";
    cudaCheck(cudaMemset(d_clusInModule.get(), 0, maxNumModules * sizeof(uint32_t)));

    cms::cuda::launch(findClus<Phase1>,
                      {blocksPerGrid, threadsPerBlock},
                      d_raw.get(),
                      d_id.get(),
                      d_x.get(),
                      d_y.get(),
                      d_moduleStart.get(),
                      d_clusInModule.get(),
                      d_moduleId.get(),
                      d_clus.get(),
                      n);
    cudaDeviceSynchronize();
    cudaCheck(cudaMemcpy(&nModules, d_moduleStart.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t nclus[maxNumModules], moduleId[nModules];
    cudaCheck(cudaMemcpy(&nclus, d_clusInModule.get(), maxNumModules * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::cout << "before charge cut found " << std::accumulate(nclus, nclus + maxNumModules, 0) << " clusters"
              << std::endl;
    for (auto i = maxNumModules; i > 0; i--)
      if (nclus[i - 1] > 0) {
        std::cout << "last module is " << i - 1 << ' ' << nclus[i - 1] << std::endl;
        break;
      }
    if (ncl != std::accumulate(nclus, nclus + maxNumModules, 0))
      std::cout << "ERROR!!!!! wrong number of cluster found" << std::endl;

    cms::cuda::launch(clusterChargeCut<Phase1>,
                      {blocksPerGrid, threadsPerBlock},
                      clusterThresholds,
                      d_id.get(),
                      d_adc.get(),
                      d_moduleStart.get(),
                      d_clusInModule.get(),
                      d_moduleId.get(),
                      d_clus.get(),
                      n);

    cudaDeviceSynchronize();
#else   // __CUDACC__
    h_moduleStart[0] = nModules;
    countModules<Phase1>(h_id.get(), h_moduleStart.get(), h_clus.get(), n);
    memset(h_clusInModule.get(), 0, maxNumModules * sizeof(uint32_t));

    findClus<Phase1>(h_raw.get(),
                     h_id.get(),
                     h_x.get(),
                     h_y.get(),
                     h_moduleStart.get(),
                     h_clusInModule.get(),
                     h_moduleId.get(),
                     h_clus.get(),
                     n);

    nModules = h_moduleStart[0];
    auto nclus = h_clusInModule.get();

    std::cout << "before charge cut found " << std::accumulate(nclus, nclus + maxNumModules, 0) << " clusters"
              << std::endl;
    for (auto i = maxNumModules; i > 0; i--)
      if (nclus[i - 1] > 0) {
        std::cout << "last module is " << i - 1 << ' ' << nclus[i - 1] << std::endl;
        break;
      }
    if (ncl != std::accumulate(nclus, nclus + maxNumModules, 0))
      std::cout << "ERROR!!!!! wrong number of cluster found" << std::endl;

    clusterChargeCut<Phase1>(clusterThresholds,
                             h_id.get(),
                             h_adc.get(),
                             h_moduleStart.get(),
                             h_clusInModule.get(),
                             h_moduleId.get(),
                             h_clus.get(),
                             n);
#endif  // __CUDACC__

    std::cout << "found " << nModules << " Modules active" << std::endl;

#ifdef __CUDACC__
    cudaCheck(cudaMemcpy(h_id.get(), d_id.get(), size16, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_clus.get(), d_clus.get(), size32, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(&nclus, d_clusInModule.get(), maxNumModules * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(&moduleId, d_moduleId.get(), nModules * sizeof(uint32_t), cudaMemcpyDeviceToHost));
#endif  // __CUDACC__

    std::set<unsigned int> clids;
    for (int i = 0; i < n; ++i) {
      assert(h_id[i] != 666);  // only noise
      if (h_id[i] == invalidModuleId)
        continue;
      assert(h_clus[i] >= 0);
      assert(h_clus[i] < int(nclus[h_id[i]]));
      clids.insert(h_id[i] * 1000 + h_clus[i]);
      // clids.insert(h_clus[i]);
    }

    // verify no hole in numbering
    auto p = clids.begin();
    auto cmid = (*p) / 1000;
    assert(0 == (*p) % 1000);
    auto c = p;
    ++c;
    std::cout << "first clusters " << *p << ' ' << *c << ' ' << nclus[cmid] << ' ' << nclus[(*c) / 1000] << std::endl;
    std::cout << "last cluster " << *clids.rbegin() << ' ' << nclus[(*clids.rbegin()) / 1000] << std::endl;
    for (; c != clids.end(); ++c) {
      auto cc = *c;
      auto pp = *p;
      auto mid = cc / 1000;
      auto pnc = pp % 1000;
      auto nc = cc % 1000;
      if (mid != cmid) {
        assert(0 == cc % 1000);
        assert(nclus[cmid] - 1 == pp % 1000);
        // if (nclus[cmid]-1 != pp%1000) std::cout << "error size " << mid << ": "  << nclus[mid] << ' ' << pp << std::endl;
        cmid = mid;
        p = c;
        continue;
      }
      p = c;
      // assert(nc==pnc+1);
      if (nc != pnc + 1)
        std::cout << "error " << mid << ": " << nc << ' ' << pnc << std::endl;
    }

    std::cout << "found " << std::accumulate(nclus, nclus + maxNumModules, 0) << ' ' << clids.size() << " clusters"
              << std::endl;
    for (auto i = maxNumModules; i > 0; i--)
      if (nclus[i - 1] > 0) {
        std::cout << "last module is " << i - 1 << ' ' << nclus[i - 1] << std::endl;
        break;
      }
    // << " and " << seeds.size() << " seeds" << std::endl;
  }  /// end loop kkk
  return 0;
}
