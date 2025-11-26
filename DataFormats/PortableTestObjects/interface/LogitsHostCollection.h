#ifndef DataFormats_PortableTestObjects_interface_LogitsHostCollection_h
#define DataFormats_PortableTestObjects_interface_LogitsHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/LogitsSoA.h"

namespace portabletest {

  using LogitsHostCollection = PortableHostCollection<LogitsSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_LogitsHostCollection_h
