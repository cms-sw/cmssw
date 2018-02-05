#include "cuda/api_wrappers.h"

#include <iostream>
#include <iomanip>

#include <cstdint>
#include <memory>
#include <algorithm>
#include<cmath>
#include<cassert>

#include<set>
#include<vector>


constexpr uint32_t numRowsInRoc     = 80;
constexpr uint32_t numColsInRoc     = 52;

constexpr uint32_t numRowsInModule  = 2*80;
constexpr uint32_t numColsInModule  = 8*52;
constexpr uint32_t numPixsInModule  = numRowsInModule*numColsInModule;


constexpr uint32_t numColsInHalfModule  = 4*52;
constexpr uint32_t numPixsInHalfModule  = numRowsInModule*numColsInHalfModule;


constexpr uint32_t MaxNumModules = 2000;

constexpr uint32_t MaxNumPixels = 128*4000;


__global__ void countModules(uint32_t const * id,
			     uint32_t * moduleStart,
			     int32_t * clus,
			     int numElements){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= numElements) return;
  clus[i]=i;
  if (id[i]==0) return;
  auto j=i-1;
  while(j>=0 && id[j]==0) --j;
  if(j<0 || id[j]!=id[i]) {
    // boundary...
    auto loc = atomicInc(moduleStart,MaxNumModules);
    moduleStart[loc+1]=i;
  }
}


__global__ void findClus(uint32_t const * id,
			 uint16_t const * x,
			 uint16_t const * y,
			 uint8_t const * adc,
			 uint32_t const * moduleStart,
			 int32_t * clus, uint32_t * debug, 
			 int numElements){

  __shared__ bool go;
  __shared__ int nclus;

 
  auto first = moduleStart[1 + blockIdx.x];  

  auto me = id[first];

  first+=threadIdx.x;
  
  go=true;
  nclus=0;
  __syncthreads();

  
  while (go) {
   __syncthreads();
    go=false;
    __syncthreads();

    for (int i=first; i<numElements; i+=blockDim.x) {
      if (id[i]==0) continue;  // not valid
      if (id[i]!=me) break;  // end of module
      ++debug[i];
      for (int j=i+1; j<numElements; ++j) {
	if (id[j]==0) continue;  // not valid
	if (id[j]!=me) break;  // end of module
	// if (clus[i]<i) continue; // already clusterized
	if (std::abs(int(x[j])-int(x[i]))>1) continue;
	if (std::abs(int(y[j])-int(y[i]))>1) continue;
	auto old = atomicMin(&clus[j],clus[i]);
	if (old!=clus[i]) go=true;
	atomicMin(&clus[i],old);
      }
    }
  
    __syncthreads();
  }

  /*
  // fast count (nice but not much useful)
  auto laneId = threadIdx.x & 0x1f;

  for (int i=first; i<numElements; i+=blockDim.x) {
    if (id[i]==0) continue;  // not valid
    if (id[i]!=me) break;  // end of module
    auto value = clus[i]^i;
    auto mask = __ballot_sync(0xffffffff,value==0);
    if (laneId==0) atomicAdd(&nclus,__popc(mask));
  }
  
  __syncthreads();
  if (me<30)
  if (laneId==0) printf("%d clusters in module %d\n",nclus,me);
  */

   nclus=0;
  __syncthreads();
  for (int i=first; i<numElements; i+=blockDim.x) {
    if (id[i]==0) continue;  // not valid
    if (id[i]!=me) break;  // end of module
    if (clus[i]==i) {
      auto old = atomicAdd(&nclus,1);
      clus[i]=-(old+1);
    }
  }

   __syncthreads();

  for (int i=first; i<numElements; i+=blockDim.x) {
    if (id[i]==0) continue;  // not valid
    if (id[i]!=me) break;  // end of module
    if (clus[i]>=0) clus[i]=clus[clus[i]];
  }

  __syncthreads();
  for (int i=first; i<numElements; i+=blockDim.x) {
    if (id[i]==0) {clus[i]=nclus+1; continue; } // not valid
    if (id[i]!=me) break;  // end of module
    clus[i] = -clus[i] -1;
  }

  
  __syncthreads();
  if (me<30)
  if (threadIdx.x==0) printf("%d clusters in module %d\n",nclus,me);

}


int main(void)
{
  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }
  
  int numElements = 200000;
 
  
  auto h_id = std::make_unique<uint32_t[]>(numElements);
  auto h_x = std::make_unique<uint16_t[]>(numElements);
  auto h_y = std::make_unique<uint16_t[]>(numElements);
  auto h_adc = std::make_unique<uint8_t[]>(numElements);
  auto h_clus = std::make_unique<int[]>(numElements);
  
  auto h_debug = std::make_unique<unsigned int[]>(numElements);
 
  
  auto current_device = cuda::device::current::get();
  auto d_id = cuda::memory::device::make_unique<uint32_t[]>(current_device, numElements);
  auto d_x = cuda::memory::device::make_unique<uint16_t[]>(current_device, numElements);
  auto d_y = cuda::memory::device::make_unique<uint16_t[]>(current_device, numElements);
  auto d_adc = cuda::memory::device::make_unique<uint8_t[]>(current_device, numElements);
  
  auto d_clus = cuda::memory::device::make_unique<int[]>(current_device, numElements);

  auto d_moduleStart = cuda::memory::device::make_unique<uint32_t[]>(current_device, MaxNumModules+1);
  
  auto d_debug = cuda::memory::device::make_unique<unsigned int[]>(current_device, numElements);

  
    // later random number
  int n=0;
  int ncl=0;
  int y[10]={5,7,9,1,3,0,4,8,2,6};

  {
    // isolated
    int id = 1;
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
    h_id[n++]=0; // error
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
  

  for(int id=11; id<=10000; id+=10) {
    if (id/100%2) h_id[n++]=0;  // error
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
	  if (id==51)  {h_id[n++]=0; h_id[n++]=0; }// error
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
  
  cuda::memory::copy(d_id.get(), h_id.get(), size32);
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

  cuda::launch(
	       findClus,
	       { blocksPerGrid, threadsPerBlock },
	       d_id.get(), d_x.get(), d_y.get(),  d_adc.get(),
	       d_moduleStart.get(),
	       d_clus.get(), d_debug.get(),
	       n
	       );

  cuda::memory::copy(h_clus.get(), d_clus.get(), size32);
  cuda::memory::copy(h_debug.get(), d_debug.get(), size32);

  auto p = std::minmax_element(h_debug.get(),h_debug.get()+n);
  std::cout << "debug " << *p.first << ' ' << *p.second << std::endl;  

  std::set<unsigned int> clids;
  std::vector<unsigned int> seeds;
  for (int i=0; i<n; ++i) {
    if (h_id[i]==0) continue;
    clids.insert(h_id[i]*100+h_clus[i]);
		 // clids.insert(h_clus[i]);
    if (h_clus[i]==i) seeds.push_back(i);
  }
   
  std::cout << "found " << clids.size() << " clusters and " << seeds.size() << " seeds" << std::endl;



  return 0;

}

