#include<random>
#include<vector>
#include<cstdint>
#include<cmath>

#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracks.h"
using namespace  gpuVertexFinder;
#include <cuda/api_wrappers.h>


struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t>  itrack;
  std::vector<float> ztrack;
  std::vector<float> eztrack;
  std::vector<float> pttrack;
  std::vector<uint16_t> ivert;
};

struct ClusterGenerator {

  explicit ClusterGenerator(float nvert, float ntrack) :
    rgen(-13.,13), errgen(0.005,0.025), clusGen(nvert), trackGen(ntrack), gauss(0.,1.), ptGen(1.)
  {}

  void operator()(Event & ev) {

    int nclus = clusGen(reng);
    ev.zvert.resize(nclus);
    ev.itrack.resize(nclus);
    for (auto & z : ev.zvert) { 
       z = 3.5f*gauss(reng);
    }

    ev.ztrack.clear(); 
    ev.eztrack.clear();
    ev.ivert.clear();
    for (int iv=0; iv<nclus; ++iv) {
      auto nt = trackGen(reng);
      ev.itrack[nclus] = nt;
      for (int it=0; it<nt; ++it) {
       auto err = errgen(reng); // reality is not flat....
       ev.ztrack.push_back(ev.zvert[iv]+err*gauss(reng));
       ev.eztrack.push_back(err*err);
       ev.ivert.push_back(iv);
       ev.pttrack.push_back( (iv==5? 1.f:0.5f) + ptGen(reng) );
       ev.pttrack.back()*=ev.pttrack.back();
      }
    }
    // add noise
    auto nt = 2*trackGen(reng);
    for (int it=0; it<nt; ++it) {
      auto err = 0.03f;
      ev.ztrack.push_back(rgen(reng));
      ev.eztrack.push_back(err*err);
      ev.ivert.push_back(9999);
      ev.pttrack.push_back( 0.5f + ptGen(reng) );
      ev.pttrack.back()*=ev.pttrack.back();
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


#include<iostream>

int main() {

  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get();

  auto zt_d = cuda::memory::device::make_unique<float[]>(current_device, 64000);
  auto ezt2_d = cuda::memory::device::make_unique<float[]>(current_device, 64000);
  auto ptt2_d = cuda::memory::device::make_unique<float[]>(current_device, 64000);
  auto zv_d = cuda::memory::device::make_unique<float[]>(current_device, 256);
  auto wv_d = cuda::memory::device::make_unique<float[]>(current_device, 256);
  auto chi2_d = cuda::memory::device::make_unique<float[]>(current_device, 256);
  auto ptv2_d = cuda::memory::device::make_unique<float[]>(current_device, 256);
  auto ind_d = cuda::memory::device::make_unique<uint16_t[]>(current_device, 256);

  auto izt_d = cuda::memory::device::make_unique<uint8_t[]>(current_device, 64000);
  auto nn_d = cuda::memory::device::make_unique<int32_t[]>(current_device, 64000);
  auto iv_d = cuda::memory::device::make_unique<int32_t[]>(current_device, 64000);

  auto nv_d = cuda::memory::device::make_unique<uint32_t[]>(current_device, 1);
 
  auto onGPU_d = cuda::memory::device::make_unique<OnGPU[]>(current_device, 1);

  OnGPU onGPU;

  onGPU.zt = zt_d.get();
  onGPU.ezt2 = ezt2_d.get();
  onGPU.ptt2 = ptt2_d.get();
  onGPU.zv = zv_d.get();
  onGPU.wv = wv_d.get();
  onGPU.chi2 = chi2_d.get();
  onGPU.ptv2 = ptv2_d.get();
  onGPU.sortInd = ind_d.get();
  onGPU.nv = nv_d.get();
  onGPU.izt = izt_d.get();
  onGPU.nn = nn_d.get();
  onGPU.iv = iv_d.get();


  cuda::memory::copy(onGPU_d.get(), &onGPU, sizeof(OnGPU));


  Event  ev;

  for (int nav=30;nav<80;nav+=20){ 

  ClusterGenerator gen(nav,10);

  for (int i=8; i<20; ++i) {

  auto  kk=i/4;  // M param

  gen(ev);
  
  std::cout << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;

  cuda::memory::copy(onGPU.zt,ev.ztrack.data(),sizeof(float)*ev.ztrack.size());
  cuda::memory::copy(onGPU.ezt2,ev.eztrack.data(),sizeof(float)*ev.eztrack.size());
  cuda::memory::copy(onGPU.ptt2,ev.pttrack.data(),sizeof(float)*ev.eztrack.size());

  float eps = 0.1f;
  
  std::cout << "M eps " << kk << ' ' << eps << std::endl;
  
  if ( (i%4) == 0 )
    cuda::launch(clusterTracks,
		 { 1, 512+256 },
		 ev.ztrack.size(), onGPU_d.get(),kk,eps,
		 0.02f,12.0f
		 );
  
  if ( (i%4) == 1 )
    cuda::launch(clusterTracks,
		 { 1, 512+256 },
		 ev.ztrack.size(), onGPU_d.get(),kk,eps,
		 0.02f,9.0f
		 );
  
  if ( (i%4) == 2 )
    cuda::launch(clusterTracks,
		 { 1, 512+256 },
		 ev.ztrack.size(), onGPU_d.get(),kk,eps,
		 0.01f,9.0f
		 );
  
  if ( (i%4) == 3 )
    cuda::launch(clusterTracks,
		 { 1, 512+256 },
		 ev.ztrack.size(), onGPU_d.get(),kk,0.7f*eps,
		 0.01f,9.0f
		 );
  
  cudaDeviceSynchronize();
  cuda::launch(sortByPt2,
               { 1, 256 },
               ev.ztrack.size(), onGPU_d.get()
              );

  uint32_t nv;
  cuda::memory::copy(&nv, onGPU.nv, sizeof(uint32_t));

  if (nv==0) {
    std::cout << "NO VERTICES???" << std::endl;
    continue;
  }

  float zv[nv];
  float	wv[nv];
  float	chi2[nv];
  float ptv2[nv];
  int32_t nn[nv];
  uint16_t ind[nv];
  cuda::memory::copy(&zv, onGPU.zv, nv*sizeof(float));
  cuda::memory::copy(&wv, onGPU.wv, nv*sizeof(float));
  cuda::memory::copy(&chi2, onGPU.chi2, nv*sizeof(float));
  cuda::memory::copy(&ptv2, onGPU.ptv2, nv*sizeof(float));
  cuda::memory::copy(&nn, onGPU.nn, nv*sizeof(int32_t));
  cuda::memory::copy(&ind, onGPU.sortInd, nv*sizeof(uint16_t));
  for (auto j=0U; j<nv; ++j) if (nn[j]>0) chi2[j]/=float(nn[j]); 
   
  {
    auto mx = std::minmax_element(wv,wv+nv);
    std::cout << "min max error " << 1./std::sqrt(*mx.first) << ' ' <<  1./std::sqrt(*mx.second) << std::endl;
  }
  {
    auto mx = std::minmax_element(chi2,chi2+nv);
    std::cout << "min max chi2 " << *mx.first << ' ' <<  *mx.second << std::endl;
  }
  {
    auto mx = std::minmax_element(ptv2,ptv2+nv);
    std::cout << "min max ptv2 " << *mx.first << ' ' <<  *mx.second << std::endl;
    std::cout << "min max ptv2 " << ptv2[ind[0]] << ' ' <<  ptv2[ind[nv-1]] << " at "  << ind[0] << ' ' << ind[nv-1] << std::endl;

  }  

  float dd[nv];
  uint32_t ii=0;
  for (auto zr : zv) {
   auto md=500.0f;
   for (auto zs : ev.ztrack) { 
     auto d = std::abs(zr-zs);
     md = std::min(d,md);
   }
   dd[ii++] = md;
  }
  assert(ii==nv);
  if (i==6) {
    for (auto d:dd) std::cout << d << ' ';
    std::cout << std::endl;
  }
  auto mx = std::minmax_element(dd,dd+nv);
  float rms=0;
  for (auto d:dd) rms+=d*d; rms = std::sqrt(rms)/(nv-1);
  std::cout << "min max rms " << *mx.first << ' ' << *mx.second << ' ' << rms << std::endl;

  } // loop on events
  } // lopp on ave vert
  
  return 0;
}
