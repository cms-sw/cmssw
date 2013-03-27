
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef STACKED_TRACKER_GEOMETRY_BUILDER_H
#define STACKED_TRACKER_GEOMETRY_BUILDER_H

#include "DataFormats/DetId/interface/DetId.h"

#include<map>

class StackedTrackerGeometry;
class TrackerGeometry;

class StackedTrackerGeometryBuilder {
public:

  typedef std::map < uint32_t , double >	lad_to_phi_map;
  typedef std::map < uint32_t , int >		lad_to_phi_i_map;
  typedef std::map < uint32_t , uint32_t >	lad_to_lad_map;
  typedef std::map < DetId , DetId  >		detid_to_detid_map;
  typedef std::map < uint32_t , uint32_t >	lay_to_stack_i_map;

  StackedTrackerGeometry* build(const TrackerGeometry* theTracker,
				double radial_window,
				double phi_window,
				double z_window,
				int truncation_precision,
				bool makeDebugFile = false );
  

};

#endif

