/*! \class   StackedTrackerGeometryBuilder
 *  \brief   TrackerGeometry-derived class for Pt modules
 *  \details
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \author Ivan Reid
 *  \date   2008
 *
 */

#ifndef STACKED_TRACKER_GEOMETRY_BUILDER_H
#define STACKED_TRACKER_GEOMETRY_BUILDER_H

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include<map>
#include <vector>

class StackedTrackerGeometry;
class TrackerGeometry;

class StackedTrackerGeometryBuilder
{
  public:

    typedef std::map < uint32_t, double >   lad_to_phi_map;
    typedef std::map < uint32_t, int >      lad_to_phi_i_map;
    typedef std::map < uint32_t, uint32_t > lad_to_lad_map;
    typedef std::map < DetId, DetId  >      detid_to_detid_map;
    typedef std::map < uint32_t, uint32_t > lay_to_stack_i_map;

    /// "Normal" builder
    StackedTrackerGeometry* build( const TrackerGeometry* theTracker,
                                   double radial_window,
                                   double phi_window,
                                   double z_window,
                                   int truncation_precision,
                                   bool makeDebugFile = false );
  
    /// CBC3 emulation builder
    StackedTrackerGeometry* build( const TrackerGeometry* theTracker,
                                   double radial_window,
                                   double phi_window,
                                   double z_window,
                                   int truncation_precision,
                                   int theNumPartitions,
                                   unsigned theMaxStubs,
                                   std::vector< double > BarrelCut,
                                   std::vector< std::vector< double > > RingCut,
                                   bool makeDebugFile = false );

  private:
    std::vector< std::vector< int > > makeOffsetArray( double ratio,
                                                       PixelGeomDetUnit* pix0,
                                                       int numPartitions );

};

#endif

