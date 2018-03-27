#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"


#include "cuda/api_wrappers.h"

#include <iostream>
#include <iomanip>

#include <cstdint>
#include <memory>
#include <algorithm>
#include<numeric>
#include<cmath>
#include<cassert>

#include<set>
#include<vector>



  
int main(void)
{
  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }


  using namespace gpuClustering;
  
  int numElements = MaxNumPixels;
 

  // these in reality are already on GPU
  auto h_id = std::make_unique<uint16_t[]>(numElements);
  auto h_x = std::make_unique<uint16_t[]>(numElements);
  auto h_y = std::make_unique<uint16_t[]>(numElements);
  auto h_adc = std::make_unique<uint16_t[]>(numElements);

  auto h_clus = std::make_unique<int[]>(numElements);
  
  auto h_debug = std::make_unique<unsigned int[]>(numElements);
 
  
  auto current_device = cuda::device::current::get();
  auto d_id = cuda::memory::device::make_unique<uint16_t[]>(current_device, numElements);
  auto d_x = cuda::memory::device::make_unique<uint16_t[]>(current_device, numElements);
  auto d_y = cuda::memory::device::make_unique<uint16_t[]>(current_device, numElements);
  auto d_adc = cuda::memory::device::make_unique<uint16_t[]>(current_device, numElements);
  
  auto d_clus = cuda::memory::device::make_unique<int[]>(current_device, numElements);

  auto d_moduleStart = cuda::memory::device::make_unique<uint32_t[]>(current_device, MaxNumModules+1);

  auto d_clusInModule = cuda::memory::device::make_unique<uint32_t[]>(current_device, MaxNumModules);
  auto d_moduleId = cuda::memory::device::make_unique<uint32_t[]>(current_device, MaxNumModules);
  
  auto d_debug = cuda::memory::device::make_unique<unsigned int[]>(current_device, numElements);

  
    // later random number
  int n=0;
  int ncl=0;
  int y[10]={5,7,9,1,3,0,4,8,2,6};

  {
    // isolated
    int id = 42;
    int x = 10;
    ++ncl;
    h_id[n]=id;
    h_x[n]=x;
    h_y[n]=x;
    h_adc[n]=100;
    ++n;
    // diagonal
    ++ncl;
    for (int x=20; x<25; ++x) {
      h_id[n]=id;
      h_x[n]=x;
      h_y[n]=x;
      h_adc[n]=100;
      ++n;
    }
    ++ncl;
    // reversed
    for (int x=45; x>40; --x) {
      h_id[n]=id;
      h_x[n]=x;
      h_y[n]=x;
      h_adc[n]=100;
      ++n;
    }
    ++ncl;
    h_id[n++]=InvId; // error
    // messy
    int xx[5] = {21,25,23,24,22};
    for (int k=0; k<5; ++k) {
      h_id[n]=id;
      h_x[n]=xx[k];
      h_y[n]=20+xx[k];
      h_adc[n]=100;
      ++n;
    }
    // holes
    ++ncl;
    for (int k=0; k<5; ++k) {
      h_id[n]=id;
      h_x[n]=xx[k];
      h_y[n]=100;
      h_adc[n]=100;
      ++n;
      if (xx[k]%2==0) {
	h_id[n]=id;
	h_x[n]=xx[k];
	h_y[n]=101;
	h_adc[n]=100;
      ++n;
      }
    }
  }
  {
   // id == 0 (make sure it works!
    int id = 0;
    int x = 10;
    ++ncl;
    h_id[n]=id;
    h_x[n]=x;
    h_y[n]=x;
    h_adc[n]=100;
    ++n;    
  }

  
  // all odd id
  for(int id=11; id<=1800; id+=2) {
    if ( (id/20)%2) h_id[n++]=InvId;  // error
    for (int x=0; x<40; x+=4) {	 
      ++ncl;
      if ((id/10)%2) {
	for (int k=0; k<10; ++k) {
	  h_id[n]=id;
	  h_x[n]=x;
	  h_y[n]=x+y[k];
	  h_adc[n]=100;
	  ++n;
	  h_id[n]=id;
	  h_x[n]=x+1;
	  h_y[n]=x+y[k]+2;
	  h_adc[n]=100;
	  ++n;
	}
      } else {
	for (int k=0; k<10; ++k) {
	  h_id[n]=id;
	  h_x[n]=x;
	  h_y[n]=x+y[9-k];
	  h_adc[n]=100;
	  ++n;
	  if (y[k]==3) continue; // hole
	  if (id==51)  {h_id[n++]=InvId; h_id[n++]=InvId; }// error
	  h_id[n]=id;
	  h_x[n]=x+1;
	  h_y[n]=x+y[k]+2;
	  h_adc[n]=100;
	  ++n;
	}
      }
    }
  }
  std::cout << "created " << n << " digis in " << ncl << " clusters" << std::endl;
  assert(n<=numElements);
  size_t size32 = n * sizeof(unsigned int);
  size_t size16 = n * sizeof(unsigned short);
  size_t size8 = n * sizeof(uint8_t);
	 
  uint32_t nModules=0;
  cuda::memory::copy(d_moduleStart.get(),&nModules,sizeof(uint32_t));
  
  cuda::memory::copy(d_id.get(), h_id.get(), size16);
  cuda::memory::copy(d_x.get(), h_x.get(), size16);
  cuda::memory::copy(d_y.get(), h_y.get(), size16);
  cuda::memory::copy(d_adc.get(), h_adc.get(), size8);
  cuda::memory::device::zero(d_debug.get(),size32);  
  // Launch CUDA Kernels

  
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  std::cout
    << "CUDA countModules kernel launch with " << blocksPerGrid
    << " blocks of " << threadsPerBlock << " threads\n";
		     
  cuda::launch(
	       countModules, 
	       { blocksPerGrid, threadsPerBlock },
	       d_id.get(), d_moduleStart.get() ,d_clus.get(),n
	       );
  
  cuda::memory::copy(&nModules,d_moduleStart.get(),sizeof(uint32_t));
		     
  std::cout << "found " << nModules << " Modules active" << std::endl;

  
  threadsPerBlock = 256;
  blocksPerGrid = nModules;



  std::cout
    << "CUDA findModules kernel launch with " << blocksPerGrid
    << " blocks of " << threadsPerBlock << " threads\n";

  cuda::memory::device::zero(d_clusInModule.get(),MaxNumModules*sizeof(uint32_t));

  cuda::launch(
	       findClus,
	       { blocksPerGrid, threadsPerBlock },
	       d_id.get(), d_x.get(), d_y.get(),  d_adc.get(),
	       d_moduleStart.get(),
	       d_clusInModule.get(), d_moduleId.get(),
	       d_clus.get(),
	       d_debug.get(),
	       n
	       );


  uint32_t nclus[MaxNumModules], moduleId[nModules];  
  cuda::memory::copy(h_clus.get(), d_clus.get(), size32);
  cuda::memory::copy(&nclus,d_clusInModule.get(),MaxNumModules*sizeof(uint32_t));
  cuda::memory::copy(&moduleId,d_moduleId.get(),nModules*sizeof(uint32_t));

  
  cuda::memory::copy(h_debug.get(), d_debug.get(), size32);

  auto p = std::minmax_element(h_debug.get(),h_debug.get()+n);
  std::cout << "debug " << *p.first << ' ' << *p.second << std::endl;  


  
  
  std::set<unsigned int> clids;
  std::vector<unsigned int> seeds;
  for (int i=0; i<n; ++i) {
    if (h_id[i]==InvId) continue;
    assert(h_clus[i]>=0);
    assert(h_clus[i]<nclus[h_id[i]]);
    clids.insert(h_id[i]*100+h_clus[i]);
		 // clids.insert(h_clus[i]);
    // if (h_clus[i]==i) seeds.push_back(i); // only if no renumbering
  }
   
  std::cout << "found " << std::accumulate(nclus,nclus+MaxNumModules,0) << ' ' <<  clids.size() << " clusters" << std::endl;
  // << " and " << seeds.size() << " seeds" << std::endl;



  return 0;

}

