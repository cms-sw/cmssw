#include <cassert>
#include <iostream>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

int main()
{
  int rc = -1;
  try
  {
    edm::ActivityRegistry ar;

    // Test setup: check if a simple CUDA runtime API call fails:
    // if so, skip the test with the CUDAService enabled
    int deviceCount = 0;
    bool configEnabled( true );
    auto ret = cudaGetDeviceCount( &deviceCount );

    // Enable the service only if there are CUDA capable GPUs installed
    if( ret != cudaSuccess )
    {
      std::cout << "=== Test #1: SKIPPED. Unable to query the CUDA capable devices from the CUDA runtime API: ("
		<< ret << ") " << cudaGetErrorString( ret ) 
		<< ". Is the host equipped with CUDA capable GPUs? ===" << std::endl;
    } else
    {
      std::cout << "=== Test #1: CUDAService enabled only if there are CUDA capable GPUs installed. ===" << std::endl;
      // Now all runtime API calls should work:
      // a CUDA error marks the test as failed.
      deviceCount = 0;
      ret = cudaGetDeviceCount( &deviceCount );
      if( ret != cudaSuccess )
      {
	std::ostringstream errstr;
	errstr << "Unable to query the CUDA capable devices from the CUDA runtime API: ("
	       << ret << ") " << cudaGetErrorString( ret );
	throw cms::Exception( "CUDAService", errstr.str() );
      }

      // No need to skip the test if no CUDA capable devices are seen by the runtime API:
      // in that case, cudaGetDeviceCount returns error code cudaErrorNoDevice.
      configEnabled = bool( deviceCount );
      edm::ParameterSet ps;
      ps.addUntrackedParameter( "enabled", configEnabled );
      CUDAService cs( ps, ar );

      // Test that the service is enabled
      assert( cs.enabled() == configEnabled );
      std::cout << "The CUDAService is enabled." << std::endl;

      // At this point, we can get, as info, the driver and runtime versions.
      int driverVersion = 0, runtimeVersion = 0;
      ret = cudaDriverGetVersion( &driverVersion );
      if( ret != cudaSuccess )
      {
	std::ostringstream errstr;
	errstr << "Unable to query the CUDA driver version from the CUDA runtime API: ("
	       << ret << ") " << cudaGetErrorString( ret );
	throw cms::Exception( "CUDAService", errstr.str() );
      }
      ret = cudaRuntimeGetVersion( &runtimeVersion );
      if( ret != cudaSuccess )
      {
	std::ostringstream errstr;
	errstr << "Unable to query the CUDA runtime API version: ("
	       << ret << ") " << cudaGetErrorString( ret );
	throw cms::Exception( "CUDAService", errstr.str() );
      }

      std::cout << "CUDA Driver Version / Runtime Version: " << driverVersion/1000 << "." << (driverVersion%100)/10
		<< " / " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;

      // Test that the number of devices found by the service
      // is the same as detected by the CUDA runtime API
      assert( cs.numberOfDevices() == deviceCount );
      std::cout << "Detected " << cs.numberOfDevices() << " CUDA Capable device(s)" << std::endl;

      // Test that the compute capabilities of each device
      // are the same as detected by the CUDA runtime API
      for( int i=0; i<deviceCount; ++i )
      {
	cudaDeviceProp deviceProp;
	ret = cudaGetDeviceProperties( &deviceProp, i );
	if( ret != cudaSuccess )
	{
	  std::ostringstream errstr;
	  errstr << "Unable to query the CUDA properties for device " << i << " from the CUDA runtime API: ("
		 << ret << ") " << cudaGetErrorString( ret );
	  throw cms::Exception( "CUDAService", errstr.str() );
	  }

	assert(deviceProp.major == cs.computeCapability(i).first);
	assert(deviceProp.minor == cs.computeCapability(i).second);
	std::cout << "Device " << i << ": " << deviceProp.name
		  << "\n CUDA Capability Major/Minor version number: " << deviceProp.major << "." << deviceProp.minor
		  << std::endl;
	std::cout << std::endl;
      }
    }
    std::cout << "=== END Test #1. ===\n" << std::endl;

    // Now forcing the service to be disabled...
    std::cout << "=== Test #2: CUDAService forced to be disabled. ===" << std::endl;
    edm::ParameterSet psf;
    configEnabled = false;
    psf.addUntrackedParameter( "enabled", configEnabled );
    CUDAService csf( psf, ar );
    std::cout << "CUDAService disabled by configuration." << std::endl;

    // Test that the service is actually disabled
    assert( csf.enabled() == configEnabled );
    assert( csf.numberOfDevices() == 0 );
    std::cout << "=== END Test #2. ===\n" << std::endl;

    //Fake the end-of-job signal.
    ar.postEndJobSignal_();
    rc = 0;
  }
  catch( cms::Exception & exc )
  {
    std::cerr << "*** CMS Exception caught. ***" << std::endl;
    std::cerr << exc << std::endl;
    rc = 1;
  }
  catch( ... )
  {
    std::cerr << "Unknown exception caught." << std::endl;
    rc = 2;
  }
  return rc;
}
