#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"
// PixelTrackUtilities only included in order to compile SoALayout with Eigen columns
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousDevice.h"

#include "RecoPixelVertexing/PixelVertexFinding/plugins/PixelVertexWorkSpaceUtilities.h"
#include "RecoPixelVertexing/PixelVertexFinding/plugins/PixelVertexWorkSpaceSoAHost.h"
#include "RecoPixelVertexing/PixelVertexFinding/plugins/PixelVertexWorkSpaceSoADevice.h"
#ifdef USE_DBSCAN
#include "RecoPixelVertexing/PixelVertexFinding/plugins/gpuClusterTracksDBSCAN.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksDBSCAN
#elif USE_ITERATIVE
#include "RecoPixelVertexing/PixelVertexFinding/plugins/gpuClusterTracksIterative.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksIterative
#else
#include "RecoPixelVertexing/PixelVertexFinding/plugins/gpuClusterTracksByDensity.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksByDensityKernel
#endif
#include "RecoPixelVertexing/PixelVertexFinding/plugins/gpuFitVertices.h"
#include "RecoPixelVertexing/PixelVertexFinding/plugins/gpuSortByPt2.h"
#include "RecoPixelVertexing/PixelVertexFinding/plugins/gpuSplitVertices.h"

#ifdef ONE_KERNEL
#ifdef __CUDACC__
__global__ void vertexFinderOneKernel(gpuVertexFinder::VtxSoAView pdata,
                                      gpuVertexFinder::WsSoAView pws,
                                      int minT,      // min number of neighbours to be "seed"
                                      float eps,     // max absolute distance to cluster
                                      float errmax,  // max error to be "seed"
                                      float chi2max  // max normalized distance to cluster,
) {
  gpuVertexFinder::clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
  __syncthreads();
  gpuVertexFinder::fitVertices(pdata, pws, 50.);
  __syncthreads();
  gpuVertexFinder::splitVertices(pdata, pws, 9.f);
  __syncthreads();
  gpuVertexFinder::fitVertices(pdata, pws, 5000.);
  __syncthreads();
  gpuVertexFinder::sortByPt2(pdata, pws);
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
    ev.pttrack.clear();
    for (int iv = 0; iv < nclus; ++iv) {
      auto nt = trackGen(reng);
      ev.itrack[iv] = nt;
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

__global__ void print(gpuVertexFinder::VtxSoAView pdata, gpuVertexFinder::WsSoAView pws) {
  auto& __restrict__ ws = pws;
  printf("nt,nv %d %d,%d\n", ws.ntrks(), pdata.nvFinal(), ws.nvIntermediate());
}

int main() {
#ifdef __CUDACC__
  cudaStream_t stream;
  cms::cudatest::requireDevices();
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  ZVertexSoADevice onGPU_d(stream);
  gpuVertexFinder::workSpace::PixelVertexWorkSpaceSoADevice ws_d(stream);
#else

  ZVertexSoAHost onGPU_d;
  gpuVertexFinder::workSpace::PixelVertexWorkSpaceSoAHost ws_d;
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
      gpuVertexFinder::init<<<1, 1, 0, stream>>>(onGPU_d.view(), ws_d.view());
#else
      gpuVertexFinder::init(onGPU_d.view(), ws_d.view());
#endif

      std::cout << "v,t size " << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;
      auto nt = ev.ztrack.size();
#ifdef __CUDACC__
      cudaCheck(cudaMemcpy(&ws_d.view().ntrks(), &nt, sizeof(uint32_t), cudaMemcpyHostToDevice));
      cudaCheck(
          cudaMemcpy(ws_d.view().zt(), ev.ztrack.data(), sizeof(float) * ev.ztrack.size(), cudaMemcpyHostToDevice));
      cudaCheck(
          cudaMemcpy(ws_d.view().ezt2(), ev.eztrack.data(), sizeof(float) * ev.eztrack.size(), cudaMemcpyHostToDevice));
      cudaCheck(
          cudaMemcpy(ws_d.view().ptt2(), ev.pttrack.data(), sizeof(float) * ev.eztrack.size(), cudaMemcpyHostToDevice));
#else
      ::memcpy(&ws_d.view().ntrks(), &nt, sizeof(uint32_t));
      ::memcpy(ws_d.view().zt(), ev.ztrack.data(), sizeof(float) * ev.ztrack.size());
      ::memcpy(ws_d.view().ezt2(), ev.eztrack.data(), sizeof(float) * ev.eztrack.size());
      ::memcpy(ws_d.view().ptt2(), ev.pttrack.data(), sizeof(float) * ev.eztrack.size());
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
      print<<<1, 1, 0, stream>>>(onGPU_d.view(), ws_d.view());
      cudaCheck(cudaGetLastError());
      cudaDeviceSynchronize();

#ifdef ONE_KERNEL
      cms::cuda::launch(vertexFinderOneKernel, {1, 512 + 256}, onGPU_d.view(), ws_d.view(), kk, par[0], par[1], par[2]);
#else
      cms::cuda::launch(CLUSTERIZE, {1, 512 + 256}, onGPU_d.view(), ws_d.view(), kk, par[0], par[1], par[2]);
#endif
      print<<<1, 1, 0, stream>>>(onGPU_d.view(), ws_d.view());

      cudaCheck(cudaGetLastError());
      cudaDeviceSynchronize();

      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.view(), ws_d.view(), 50.f);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaMemcpy(&nv, &onGPU_d.view().nvFinal(), sizeof(uint32_t), cudaMemcpyDeviceToHost));

#else
      print(onGPU_d.view(), ws_d.view());
      CLUSTERIZE(onGPU_d.view(), ws_d.view(), kk, par[0], par[1], par[2]);
      print(onGPU_d.view(), ws_d.view());
      gpuVertexFinder::fitVertices(onGPU_d.view(), ws_d.view(), 50.f);
      nv = onGPU_d.view().nvFinal();
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
      zv = onGPU_d.view().zv();
      wv = onGPU_d.view().wv();
      ptv2 = onGPU_d.view().ptv2();
      nn = onGPU_d.view().ndof();
      ind = onGPU_d.view().sortInd();
#endif

#ifdef __CUDACC__
      cudaCheck(cudaMemcpy(nn, onGPU_d.view().ndof(), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float), cudaMemcpyDeviceToHost));
#else
      memcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float));
#endif

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "after fit nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

#ifdef __CUDACC__
      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.view(), ws_d.view(), 50.f);
      cudaCheck(cudaMemcpy(&nv, &onGPU_d.view().nvFinal(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nn, onGPU_d.view().ndof(), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float), cudaMemcpyDeviceToHost));
#else
      gpuVertexFinder::fitVertices(onGPU_d.view(), ws_d.view(), 50.f);
      nv = onGPU_d.view().nvFinal();
      memcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float));
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
      cms::cuda::launch(gpuVertexFinder::splitVerticesKernel, {1024, 64}, onGPU_d.view(), ws_d.view(), 9.f);
      cudaCheck(cudaMemcpy(&nv, &ws_d.view().nvIntermediate(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
#else
      gpuVertexFinder::splitVertices(onGPU_d.view(), ws_d.view(), 9.f);
      nv = ws_d.view().nvIntermediate();
#endif
      std::cout << "after split " << nv << std::endl;

#ifdef __CUDACC__
      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.view(), ws_d.view(), 5000.f);
      cudaCheck(cudaGetLastError());

      cms::cuda::launch(gpuVertexFinder::sortByPt2Kernel, {1, 256}, onGPU_d.view(), ws_d.view());
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaMemcpy(&nv, &onGPU_d.view().nvFinal(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
#else
      gpuVertexFinder::fitVertices(onGPU_d.view(), ws_d.view(), 5000.f);
      gpuVertexFinder::sortByPt2(onGPU_d.view(), ws_d.view());
      nv = onGPU_d.view().nvFinal();
      memcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float));
#endif

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

#ifdef __CUDACC__
      cudaCheck(cudaMemcpy(zv, onGPU_d.view().zv(), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(wv, onGPU_d.view().wv(), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(ptv2, onGPU_d.view().ptv2(), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nn, onGPU_d.view().ndof(), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(ind, onGPU_d.view().sortInd(), nv * sizeof(uint16_t), cudaMemcpyDeviceToHost));
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
