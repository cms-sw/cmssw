#ifndef DataFormats_PortableTestObjects_interface_TestSoA_h
#define DataFormats_PortableTestObjects_interface_TestSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace portabletest {

  // SoA layout with x, y, z, id fields
  GENERATE_SOA_LAYOUT(TestSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(double, x),
                      SOA_COLUMN(double, y),
                      SOA_COLUMN(double, z),
                      SOA_COLUMN(int32_t, id))

  using TestSoA = TestSoALayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestSoA_h
