#include <cassert>
#include <iostream>
#include <string>
#include <utility>

#include <cuda.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/getCudaDrvErrorString.h"

int main()
{
  int rc = -1;
  try
  {
    edm::ActivityRegistry ar;

    // Test setup: check if you can initialize the CUDA driver API:
    // if failing, skip the test with the CUDAService enabled
    bool configEnabled( true );
    auto ret = cuInit( 0 );

    // Enable the service only if there are CUDA capable GPUs installed
    if( ret != CUDA_SUCCESS )
    {
      std::cout << "=== Test #1: SKIPPED. Unable to initialize the CUDA driver API: ("
		<< ret << ") " << getCudaDrvErrorString( ret )
		<< ". Is the host equipped with CUDA capable GPUs? ===" << std::endl;
    } else
    {
      std::cout << "=== Test #1: CUDAService enabled only if there are CUDA capable GPUs installed. ===" << std::endl;
      // Now all driver API calls should work:
      // a CUDA error marks the test as failed.
      int deviceCount = 0;
      ret = cuDeviceGetCount( &deviceCount );
      if( ret != CUDA_SUCCESS )
      {
	std::ostringstream errstr;
	errstr << "Unable to query the CUDA capable devices from the CUDA driver API: ("
	       << ret << ") " << getCudaDrvErrorString( ret );
	throw cms::Exception("CUDAService", errstr.str() );
      }

      // If no CUDA capable devices are seen by the driver API, cuDeviceGetCount returns 0,
      // with error code CUDA_SUCCESS. We construct the CUDAService instance as enabled,
      // and then test if it is disabled when no CUDA capable devices are visible.
      edm::ParameterSet ps;
      ps.addUntrackedParameter( "enabled", configEnabled );
      CUDAService cs( ps, ar );

      // Test that the service is enabled only if there are visible CUDA capable devices.
      assert( cs.enabled() == bool( deviceCount ) );
      std::cout << "The CUDAService is "
                << (deviceCount ? "enabled." : "disabled.") << std::endl;

      // At this point, we can get, as info, the driver and runtime versions.
      int driverVersion = 0;
      ret = cuDriverGetVersion( &driverVersion );
      if( ret != CUDA_SUCCESS )
      {
	std::ostringstream errstr;
	errstr << "Unable to query the CUDA driver version: ("
	       << ret << ") " << getCudaDrvErrorString( ret );
	throw cms::Exception( "CUDAService", errstr.str() );
      }
      std::cout << "CUDA Driver Version: " << driverVersion/1000 << "." << (driverVersion%100)/10 << std::endl;

      // Test that the number of devices found by the service
      // is the same as detected by the CUDA runtime API
      assert( cs.numberOfDevices() == deviceCount );
      if( deviceCount > 0 )
      {
        std::cout << "Detected " << cs.numberOfDevices() << " CUDA Capable device(s)" << std::endl;
      } else
      {
	std::cout << "There are no available device(s) that support CUDA" << std::endl;
      }

      // Test that the compute capabilities of each device
      // are the same as detected by the CUDA driver API
      int major = 0, minor = 0;
      char deviceName[256];
      for( CUdevice i=0; i<deviceCount; ++i )
      {
	ret = cuDeviceComputeCapability(&major, &minor, i);
	if( ret != CUDA_SUCCESS )
	{
	  std::ostringstream errstr;
	  errstr << "Unable to query the device " << i << " compute capability using the CUDA driver API: ("
		 << ret << ") " << getCudaDrvErrorString( ret );
	  throw cms::Exception( "CUDAService", errstr.str() );
	}
        ret = cuDeviceGetName( deviceName, 256, i );
        if( ret != CUDA_SUCCESS )
	{
	  std::ostringstream errstr;
	  errstr << "Unable to query the device " << i << " name using the CUDA driver API: ("
		 << ret << ") " << getCudaDrvErrorString( ret );
	  throw cms::Exception( "CUDAService", errstr.str() );
	}
	assert(major == cs.computeCapability(i).first);
	assert(minor == cs.computeCapability(i).second);
	std::cout << "Device " << i << ": " << deviceName
		  << "\n CUDA Capability Major/Minor version number: " << major << "." << minor
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
