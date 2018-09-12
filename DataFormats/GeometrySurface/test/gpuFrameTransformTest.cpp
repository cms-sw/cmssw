#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include <cstdint>



void toGlobalWrapper(SOAFrame<float> const * frame, 
	      float const * xl, float const * yl,
	      float * x, float * y, float * z,
              float const * le, float * ge,
		     uint32_t n);



#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"


#include "cuda/api_wrappers.h"


#include <cstdint>
#include <memory>
#include <algorithm>
#include<numeric>
#include<cmath>
#include<cassert>



#include<iostream>
#include <iomanip>

int main(void)
{

  typedef float T; 
  typedef TkRotation<T>                   Rotation;
  typedef SOARotation<T>                  SRotation;
  typedef GloballyPositioned<T>           Frame;
  typedef SOAFrame<T>                     SFrame;
  typedef typename Frame::PositionType             Position;
  typedef typename Frame::GlobalVector             GlobalVector;
  typedef typename Frame::GlobalPoint              GlobalPoint;
  typedef typename Frame::LocalVector              LocalVector;
  typedef typename Frame::LocalPoint               LocalPoint;

  

  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }



  constexpr uint32_t size = 10000;
  constexpr uint32_t size32 = size*sizeof(float);


  float xl[size],yl[size];
  float x[size],y[size],z[size];

  // errors
  float le[3*size];
  float ge[6*size];

  
  auto current_device = cuda::device::current::get();
  auto d_xl = cuda::memory::device::make_unique<float[]>(current_device, size);
  auto d_yl = cuda::memory::device::make_unique<float[]>(current_device, size);
 
  auto d_x = cuda::memory::device::make_unique<float[]>(current_device, size);
  auto d_y = cuda::memory::device::make_unique<float[]>(current_device, size);
  auto d_z = cuda::memory::device::make_unique<float[]>(current_device, size);

  auto d_le = cuda::memory::device::make_unique<float[]>(current_device, 3*size);
  auto d_ge = cuda::memory::device::make_unique<float[]>(current_device, 6*size);


  double a = 0.01;
  double ca = std::cos(a);
  double sa = std::sin(a);
  
  Rotation r1(ca, sa, 0,
	      -sa, ca, 0,
	      0,   0,  1);
  Frame f1(Position(2,3,4), r1);
  std::cout << "f1.position() " << f1.position() << std::endl;
  std::cout << "f1.rotation() " << '\n' << f1.rotation() << std::endl;

  SFrame sf1(f1.position().x(),
	     f1.position().y(),
	     f1.position().z(),
	     f1.rotation()
	     );

  
  // auto d_sf = cuda::memory::device::make_unique<SFrame[]>(current_device, 1);
  auto d_sf = cuda::memory::device::make_unique<char[]>(current_device, sizeof(SFrame));
  cuda::memory::copy(d_sf.get(), &sf1, sizeof(SFrame));
		     
		     
  
  for (auto i=0U; i<size; ++i) {
    xl[i]=yl[i] =  0.1f*float(i)-float(size/2);
    le[3*i] = 0.01f; le[3*i+2] =  (i>size/2) ? 1.f :  0.04f;
    le[2*i+1]=0.;
  }
  std::random_shuffle(xl,xl+size);
  std::random_shuffle(yl,yl+size);
  
  cuda::memory::copy(d_xl.get(), xl, size32);
  cuda::memory::copy(d_yl.get(), yl, size32);
  cuda::memory::copy(d_le.get(), le, 3*size32);
  

  toGlobalWrapper((SFrame const *)(d_sf.get()), d_xl.get(), d_yl.get(), d_x.get(), d_y.get(), d_z.get(),
    d_le.get(), d_ge.get(), size
    );
  
  cuda::memory::copy(x,d_x.get(), size32);
  cuda::memory::copy(y,d_y.get(), size32);
  cuda::memory::copy(z,d_z.get(), size32);
  cuda::memory::copy(ge,d_ge.get(), 6*size32);
  

  float eps=0.;
  for (auto i=0U; i<size; ++i) {
    auto gp = f1.toGlobal(LocalPoint(xl[i],yl[i]));
    eps = std::max(eps,std::abs(x[i]-gp.x()));
    eps = std::max(eps,std::abs(y[i]-gp.y()));
    eps = std::max(eps,std::abs(z[i]-gp.z()));
  }  
  
  std::cout << "max eps " << eps << std::endl;
  
  
  return 0;
}

