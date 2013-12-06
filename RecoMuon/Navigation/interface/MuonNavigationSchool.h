#ifndef Navigation_MuonNavigationSchool_H
#define Navigation_MuonNavigationSchool_H

/** \class MuonNavigationSchool
 *
 * Description:
 *  Navigation school for the muon system
 *  This class defines which DetLayers are reacheable from each Muon DetLayer
 *  (DT, CSC and RPC). The reacheableness is based on an eta range criteria.
 *
 * $Date: 2007/06/14 20:31:43 $
 * $Revision: 1.7 $
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
    MuonNavigationSchool(const MuonDetLayerGeometry *, bool enableRPC = true, bool enableCSC = true, bool enableGEM = true);
    /// Destructor
    ~MuonNavigationSchool();
    /// return navigable layers, from base class
    virtual StateType navigableLayers() const;
  private:
    /// add barrel layer
    void addBarrelLayer(BarrelDetLayer*);
    /// add endcap layer (backward and forward)
    void addEndcapLayer(ForwardDetLayer*);
    /// link barrel layers
    void linkBarrelLayers();
    /// link endcap layers
    void linkEndcapLayers(const MapE&,std::vector<MuonForwardNavigableLayer*>&);
    /// establish inward links
    void createInverseLinks() const;
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
