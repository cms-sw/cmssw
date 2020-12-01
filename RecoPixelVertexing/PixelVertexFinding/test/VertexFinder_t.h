#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#ifdef USE_DBSCAN
#include "../plugins/gpuClusterTracksDBSCAN.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksDBSCAN
#elif USE_ITERATIVE
#include "../plugins/gpuClusterTracksIterative.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksIterative
#else
#include "../plugins/gpuClusterTracksByDensity.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksByDensityKernel
#endif
#include "../plugins/gpuFitVertices.h"
#include "../plugins/gpuSortByPt2.h"
#include "../plugins/gpuSplitVertices.h"

#ifdef ONE_KERNEL
#ifdef __CUDACC__
__global__ void vertexFinderOneKernel(gpuVertexFinder::ZVertices* pdata,
                                      gpuVertexFinder::WorkSpace* pws,
                                      int minT,      // min number of neighbours to be "seed"
                                      float eps,     // max absolute distance to cluster
                                      float errmax,  // max error to be "seed"
                                      float chi2max  // max normalized distance to cluster,
) {
  clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
  __syncthreads();
  fitVertices(pdata, pws, 50.);
  __syncthreads();
  splitVertices(pdata, pws, 9.f);
  __syncthreads();
  fitVertices(pdata, pws, 5000.);
  __syncthreads();
  sortByPt2(pdata, pws);
}
#endif
#endif

struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t> itrack;
  std::vector<float> ztrack;
  std::vector<float> eztrack;
  std::vector<float> pttrack;
  std::vector<uint16_t> ivert;
};

struct ClusterGenerator {
  explicit ClusterGenerator(float nvert, float ntrack)
      : rgen(-13., 13), errgen(0.005, 0.025), clusGen(nvert), trackGen(ntrack), gauss(0., 1.), ptGen(1.) {}

  void operator()(Event& ev) {
    int nclus = clusGen(reng);
    ev.zvert.resize(nclus);
    ev.itrack.resize(nclus);
    for (auto& z : ev.zvert) {
      z = 3.5f * gauss(reng);
    }

    ev.ztrack.clear();
    ev.eztrack.clear();
    ev.ivert.clear();
    for (int iv = 0; iv < nclus; ++iv) {
      auto nt = trackGen(reng);
      ev.itrack[nclus] = nt;
      for (int it = 0; it < nt; ++it) {
        auto err = errgen(reng);  // reality is not flat....
        ev.ztrack.push_back(ev.zvert[iv] + err * gauss(reng));
        ev.eztrack.push_back(err * err);
        ev.ivert.push_back(iv);
        ev.pttrack.push_back((iv == 5 ? 1.f : 0.5f) + ptGen(reng));
        ev.pttrack.back() *= ev.pttrack.back();
      }
    }
    // add noise
    auto nt = 2 * trackGen(reng);
    for (int it = 0; it < nt; ++it) {
      auto err = 0.03f;
      ev.ztrack.push_back(rgen(reng));
      ev.eztrack.push_back(err * err);
      ev.ivert.push_back(9999);
      ev.pttrack.push_back(0.5f + ptGen(reng));
      ev.pttrack.back() *= ev.pttrack.back();
    }
  }

  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen;
  std::uniform_real_distribution<float> errgen;
  std::poisson_distribution<int> clusGen;
  std::poisson_distribution<int> trackGen;
  std::normal_distribution<float> gauss;
  std::exponential_distribution<float> ptGen;
};

// a macro SORRY
#define LOC_ONGPU(M) ((char*)(onGPU_d.get()) + offsetof(gpuVertexFinder::ZVertices, M))
#define LOC_WS(M) ((char*)(ws_d.get()) + offsetof(gpuVertexFinder::WorkSpace, M))

__global__ void print(gpuVertexFinder::ZVertices const* pdata, gpuVertexFinder::WorkSpace const* pws) {
  auto const& __restrict__ data = *pdata;
  auto const& __restrict__ ws = *pws;
  printf("nt,nv %d %d,%d\n", ws.ntrks, data.nvFinal, ws.nvIntermediate);
}

int main() {
#ifdef __CUDACC__
  cms::cudatest::requireDevices();

  auto onGPU_d = cms::cuda::make_device_unique<gpuVertexFinder::ZVertices[]>(1, nullptr);
  auto ws_d = cms::cuda::make_device_unique<gpuVertexFinder::WorkSpace[]>(1, nullptr);
#else
  auto onGPU_d = std::make_unique<gpuVertexFinder::ZVertices>();
  auto ws_d = std::make_unique<gpuVertexFinder::WorkSpace>();
#endif

  Event ev;

  float eps = 0.1f;
  std::array<float, 3> par{{eps, 0.01f, 9.0f}};
  for (int nav = 30; nav < 80; nav += 20) {
    ClusterGenerator gen(nav, 10);

    for (int i = 8; i < 20; ++i) {
      auto kk = i / 4;  // M param

      gen(ev);

#ifdef __CUDACC__
      init<<<1, 1, 0, 0>>>(onGPU_d.get(), ws_d.get());
#else
      onGPU_d->init();
      ws_d->init();
#endif

      std::cout << "v,t size " << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;
      auto nt = ev.ztrack.size();
#ifdef __CUDACC__
      cudaCheck(cudaMemcpy(LOC_WS(ntrks), &nt, sizeof(uint32_t), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(LOC_WS(zt), ev.ztrack.data(), sizeof(float) * ev.ztrack.size(), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(LOC_WS(ezt2), ev.eztrack.data(), sizeof(float) * ev.eztrack.size(), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(LOC_WS(ptt2), ev.pttrack.data(), sizeof(float) * ev.eztrack.size(), cudaMemcpyHostToDevice));
#else
      ::memcpy(LOC_WS(ntrks), &nt, sizeof(uint32_t));
      ::memcpy(LOC_WS(zt), ev.ztrack.data(), sizeof(float) * ev.ztrack.size());
      ::memcpy(LOC_WS(ezt2), ev.eztrack.data(), sizeof(float) * ev.eztrack.size());
      ::memcpy(LOC_WS(ptt2), ev.pttrack.data(), sizeof(float) * ev.eztrack.size());
#endif

      std::cout << "M eps, pset " << kk << ' ' << eps << ' ' << (i % 4) << std::endl;

      if ((i % 4) == 0)
        par = {{eps, 0.02f, 12.0f}};
      if ((i % 4) == 1)
        par = {{eps, 0.02f, 9.0f}};
      if ((i % 4) == 2)
        par = {{eps, 0.01f, 9.0f}};
      if ((i % 4) == 3)
        par = {{0.7f * eps, 0.01f, 9.0f}};

      uint32_t nv = 0;
#ifdef __CUDACC__
      print<<<1, 1, 0, 0>>>(onGPU_d.get(), ws_d.get());
      cudaCheck(cudaGetLastError());
      cudaDeviceSynchronize();

#ifdef ONE_KERNEL
      cms::cuda::launch(vertexFinderOneKernel, {1, 512 + 256}, onGPU_d.get(), ws_d.get(), kk, par[0], par[1], par[2]);
#else
      cms::cuda::launch(CLUSTERIZE, {1, 512 + 256}, onGPU_d.get(), ws_d.get(), kk, par[0], par[1], par[2]);
#endif
      print<<<1, 1, 0, 0>>>(onGPU_d.get(), ws_d.get());

      cudaCheck(cudaGetLastError());
      cudaDeviceSynchronize();

      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.get(), ws_d.get(), 50.f);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaMemcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t), cudaMemcpyDeviceToHost));

#else
      print(onGPU_d.get(), ws_d.get());
      CLUSTERIZE(onGPU_d.get(), ws_d.get(), kk, par[0], par[1], par[2]);
      print(onGPU_d.get(), ws_d.get());
      fitVertices(onGPU_d.get(), ws_d.get(), 50.f);
      nv = onGPU_d->nvFinal;
#endif

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

      float* zv = nullptr;
      float* wv = nullptr;
      float* ptv2 = nullptr;
      int32_t* nn = nullptr;
      uint16_t* ind = nullptr;

      // keep chi2 separated...
      float chi2[2 * nv];  // make space for splitting...

#ifdef __CUDACC__
      float hzv[2 * nv];
      float hwv[2 * nv];
      float hptv2[2 * nv];
      int32_t hnn[2 * nv];
      uint16_t hind[2 * nv];

      zv = hzv;
      wv = hwv;
      ptv2 = hptv2;
      nn = hnn;
      ind = hind;
#else
      zv = onGPU_d->zv;
      wv = onGPU_d->wv;
      ptv2 = onGPU_d->ptv2;
      nn = onGPU_d->ndof;
      ind = onGPU_d->sortInd;
#endif

#ifdef __CUDACC__
      cudaCheck(cudaMemcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float), cudaMemcpyDeviceToHost));
#else
      memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));
#endif

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "after fit nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

#ifdef __CUDACC__
      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.get(), ws_d.get(), 50.f);
      cudaCheck(cudaMemcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float), cudaMemcpyDeviceToHost));
#else
      fitVertices(onGPU_d.get(), ws_d.get(), 50.f);
      nv = onGPU_d->nvFinal;
      memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));
#endif

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "before splitting nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

#ifdef __CUDACC__
      // one vertex per block!!!
      cms::cuda::launch(gpuVertexFinder::splitVerticesKernel, {1024, 64}, onGPU_d.get(), ws_d.get(), 9.f);
      cudaCheck(cudaMemcpy(&nv, LOC_WS(nvIntermediate), sizeof(uint32_t), cudaMemcpyDeviceToHost));
#else
      splitVertices(onGPU_d.get(), ws_d.get(), 9.f);
      nv = ws_d->nvIntermediate;
#endif
      std::cout << "after split " << nv << std::endl;

#ifdef __CUDACC__
      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.get(), ws_d.get(), 5000.f);
      cudaCheck(cudaGetLastError());

      cms::cuda::launch(gpuVertexFinder::sortByPt2Kernel, {1, 256}, onGPU_d.get(), ws_d.get());
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaMemcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t), cudaMemcpyDeviceToHost));
#else
      fitVertices(onGPU_d.get(), ws_d.get(), 5000.f);
      sortByPt2(onGPU_d.get(), ws_d.get());
      nv = onGPU_d->nvFinal;
      memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));
#endif

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

#ifdef __CUDACC__
      cudaCheck(cudaMemcpy(zv, LOC_ONGPU(zv), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(wv, LOC_ONGPU(wv), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(ptv2, LOC_ONGPU(ptv2), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(ind, LOC_ONGPU(sortInd), nv * sizeof(uint16_t), cudaMemcpyDeviceToHost));
#endif
      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      {
        auto mx = std::minmax_element(wv, wv + nv);
        std::cout << "min max error " << 1. / std::sqrt(*mx.first) << ' ' << 1. / std::sqrt(*mx.second) << std::endl;
      }

      {
        auto mx = std::minmax_element(ptv2, ptv2 + nv);
        std::cout << "min max ptv2 " << *mx.first << ' ' << *mx.second << std::endl;
        std::cout << "min max ptv2 " << ptv2[ind[0]] << ' ' << ptv2[ind[nv - 1]] << " at " << ind[0] << ' '
                  << ind[nv - 1] << std::endl;
      }

      float dd[nv];
      for (auto kv = 0U; kv < nv; ++kv) {
        auto zr = zv[kv];
        auto md = 500.0f;
        for (auto zs : ev.ztrack) {
          auto d = std::abs(zr - zs);
          md = std::min(d, md);
        }
        dd[kv] = md;
      }
      if (i == 6) {
        for (auto d : dd)
          std::cout << d << ' ';
        std::cout << std::endl;
      }
      auto mx = std::minmax_element(dd, dd + nv);
      float rms = 0;
      for (auto d : dd)
        rms += d * d;
      rms = std::sqrt(rms) / (nv - 1);
      std::cout << "min max rms " << *mx.first << ' ' << *mx.second << ' ' << rms << std::endl;

    }  // loop on events
  }    // lopp on ave vert

  return 0;
}
