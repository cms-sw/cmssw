#ifndef Navigation_MTDNavigationSchool_H
#define Navigation_MTDNavigationSchool_H

/** \class MTDNavigationSchool
 *
 * Description:
 *  Navigation school for the MTD system
 *  This class defines which DetLayers are reacheable from each MTD DetLayer
 *  ( BTL and ETL ). The reacheableness is based on an eta range criteria.
 *
 *
 * \author : L. Gray
 *
 * Modification:
 *
 
 */


#include "RecoMTD/Navigation/interface/MTDDetLayerMap.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include <vector>
#include <map>

class BTLNavigableLayer;
class ETLNavigableLayer;
class MTDEtaRange;
class BarrelDetLayer;
class ForwardDetLayer;

class MTDNavigationSchool : public NavigationSchool {

  public:
    ///Constructor
    MTDNavigationSchool(const MTDDetLayerGeometry *, bool enableBTL = true, bool enableETL = true);
    /// Destructor
    ~MTDNavigationSchool() override;
    /// return navigable layers, from base class
    StateType navigableLayers() override;
  private:
    /// add barrel layer
    void addBarrelLayer(const BarrelDetLayer*);
    /// add endcap layer (backward and forward)
    void addEndcapLayer(const ForwardDetLayer*);
    /// link barrel layers
    void linkBarrelLayers();
    /// link endcap layers
    void linkEndcapLayers(const MapE&,std::vector<ETLNavigableLayer*>&);
    /// establish inward links
    void createInverseLinks();
    float calculateEta(const float&, const float& ) const;

  private: 
  
    struct delete_layer
    {
      template <typename T>
      void operator()(T*& p)
      {
        if( p)
        {
          delete p;
          p = nullptr;
        }
      }
    };

    MapB theBarrelLayers;    /// barrel
    MapE theForwardLayers;   /// +z endcap
    MapE theBackwardLayers;  /// -z endcap 

    std::vector<BTLNavigableLayer*> theBarrelNLC;
    std::vector<ETLNavigableLayer*> theForwardNLC;
    std::vector<ETLNavigableLayer*> theBackwardNLC;

    const MTDDetLayerGeometry * theMTDDetLayerGeometry; 
  
};
#endif
