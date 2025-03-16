#ifndef DataFormats_HeterogeneousTutorial_interface_alpaka_TripletsDeviceCollection_h
#define DataFormats_HeterogeneousTutorial_interface_alpaka_TripletsDeviceCollection_h

#include "DataFormats/HeterogeneousTutorial/interface/TripletsHostCollection.h"
#include "DataFormats/HeterogeneousTutorial/interface/TripletsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  // Make the names from the top-level tutorial namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::tutorial namespace.
  using namespace ::tutorial;

  using TripletsDeviceCollection = PortableCollection<TripletsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

// Check that the portable device collection for the host device is the same as the portable host collection.
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(tutorial::TripletsDeviceCollection, tutorial::TripletsHostCollection);

#endif  // DataFormats_HeterogeneousTutorial_interface_alpaka_TripletsDeviceCollection_h
