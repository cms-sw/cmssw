// these first to make sure they get included before any SoA header
#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/PortableTestObjects/interface/TestProductWithPtr.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
