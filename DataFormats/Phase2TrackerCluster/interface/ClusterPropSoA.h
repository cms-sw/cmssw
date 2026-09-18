//================================== Defines the SoA data structure holding the unpacked cluster properties.====================================
#ifndef DataFormats_Phase2TrackerCluster_interface_ClusterPropSoA_h
#define DataFormats_Phase2TrackerCluster_interface_ClusterPropSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace Phase2RawToCluster {

// ----------------------------------------------------------------------
//   detId     : module ID 
//   x         : strip index 
//   y         : sensor row
//   z         : PS pixel column
//   width     : cluster width
//   isSeed    : 1 = seed sensor (2S inner strips or PS pixels), 0 = correlated sensor (2S outer strips or PS strips)
//   mip       : PS strip MIP bit (0 for pixels and 2S)
//   moduleType: 1=2S, 2=PS
//
// ----------------------------------------------------------------------

  GENERATE_SOA_LAYOUT(ClusterPropSoALayout,
                      SOA_COLUMN(uint32_t, detId),      
                      SOA_COLUMN(uint16_t, x),          
                      SOA_COLUMN(uint16_t, y),         
                      SOA_COLUMN(uint8_t,  z),          
                      SOA_COLUMN(uint8_t,  width),      
                      SOA_COLUMN(uint8_t,  isSeed),     
                      SOA_COLUMN(uint8_t,  mip),        
                      SOA_COLUMN(uint8_t,  moduleType) 
  )

  using ClusterPropSoA = ClusterPropSoALayout<>;

}  // namespace Phase2RawToCluster

#endif  // DataFormats_Phase2TrackerCluster_interface_ClusterPropSoA_h
