#ifndef TrackingTools_MuonSeedDetLayer_H
#define TrackingTools_MuonSeedDetLayer_H

/**  \class MuonSeedDetLayer
 *
 *   Muon seed detector layer
 *   Helper class for seed generation
 *
 *
 *
 *   \author  N. Neumeister          Purdue University
 */

#include <functional>

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class MuonSeedDetLayer {

  public:

    /// constructor
    MuonSeedDetLayer(const DetLayer* layer, 
                     GlobalPoint& point, 
                     int id, 
                     TrajectoryStateOnSurface& state) :
            theLayer(layer), thePoint(point), theID(id), theState(state) {}

    /// destructor
    virtual ~MuonSeedDetLayer() {}

    /// return layer
    const DetLayer* layer() const { return theLayer; }

    /// return impact point
    const GlobalPoint& point() const { return thePoint; }

    /// return layer identifyer
    int id() const { return theID; }

    /// return trajectory state on layer
    const TrajectoryStateOnSurface& state() const { return theState; }

    /// return radius of impact point
    double radius() const { return sqrt(thePoint.x()*thePoint.x()+thePoint.y()*thePoint.y()); }

    /// return z coordinate of impact point
    double z() const { return thePoint.z(); }

    /// sort
    class HigherR : std::binary_function< const MuonSeedDetLayer, const MuonSeedDetLayer, bool>
{
      public :
        bool operator()( const MuonSeedDetLayer first, const MuonSeedDetLayer second ) const {
          return first.radius() < second.radius();
        }
    };

    /// sort
    class LowerR : std::binary_function< const MuonSeedDetLayer, const MuonSeedDetLayer, bool>
{
      public :
        bool operator()( const MuonSeedDetLayer first, const MuonSeedDetLayer second ) const {
          return first.radius() > second.radius();
        }
    };

  private:

    const DetLayer*          theLayer;
    GlobalPoint              thePoint;  
    int                      theID;
    TrajectoryStateOnSurface theState;
};
#endif 

