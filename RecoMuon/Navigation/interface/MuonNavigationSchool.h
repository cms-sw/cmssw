#ifndef Navigation_MuonNavigationSchool_H
#define Navigation_MuonNavigationSchool_H

/** \class MuonNavigationSchool
 *
 * Description:
 *  Navigation school for the muon system
 *  This class defines which DetLayers are reacheable from each Muon DetLayer
 *  (DT, CSC and RPC). The reacheableness is based on an eta range criteria.
 *
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * The class links maps for nextLayers and compatibleLayers in the same time.
 *
 * Cesare Calabria:
 * GEMs implementation.
 */


#include "RecoMuon/Navigation/interface/MuonDetLayerMap.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include <vector>
#include <map>

class MuonBarrelNavigableLayer;
class MuonForwardNavigableLayer;
class MuonEtaRange;
class BarrelDetLayer;
class ForwardDetLayer;

class MuonNavigationSchool : public NavigationSchool {

  public:
    ///Constructor
    MuonNavigationSchool(const MuonDetLayerGeometry *, bool enableRPC = true, bool enableCSC = true, bool enableGEM = false);
    /// Destructor
    ~MuonNavigationSchool();
    /// return navigable layers, from base class
    virtual StateType navigableLayers() override;
  private:
    /// add barrel layer
    void addBarrelLayer(const BarrelDetLayer*);
    /// add endcap layer (backward and forward)
    void addEndcapLayer(const ForwardDetLayer*);
    /// link barrel layers
    void linkBarrelLayers();
    /// link endcap layers
    void linkEndcapLayers(const MapE&,std::vector<MuonForwardNavigableLayer*>&);
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
          p = 0;
        }
      }
    };

    MapB theBarrelLayers;    /// barrel
    MapE theForwardLayers;   /// +z endcap
    MapE theBackwardLayers;  /// -z endcap 

    std::vector<MuonBarrelNavigableLayer*> theBarrelNLC;
    std::vector<MuonForwardNavigableLayer*> theForwardNLC;
    std::vector<MuonForwardNavigableLayer*> theBackwardNLC;

    const MuonDetLayerGeometry * theMuonDetLayerGeometry; 
  
};
#endif
