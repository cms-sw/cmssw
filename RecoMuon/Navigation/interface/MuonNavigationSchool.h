#ifndef Navigation_MuonNavigationSchool_H
#define Navigation_MuonNavigationSchool_H

/** \class MuonNavigationSchool
 *
 * Description:
 *  Navigation school for the muon system
 *  This class defines which DetLayers are reacheable from each Muon DetLayer
 *  (DT, CSC and RPC). The reacheableness is based on an eta range criteria.
 *
 * $Date: 2003/12/03 07:58:23 $
 * $Revision: 1.5 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * The class links maps for nextLayers and compatibleLayers in the same time.
 *
 */


#include "RecoMuon/Navigation/interface/MuonLayerSort.h"
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
    MuonNavigationSchool(const MuonDetLayerGeometry *);
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
    void linkEndcapLayers(const MapE&,vector<MuonForwardNavigableLayer*>&);
    /// establish inward links
    void createInverseLinks() const;
    float calculateEta(const float&, const float& ) const;

  private: 
  
    MapB theBarrelLayers;    /// barrel
    MapE theForwardLayers;   /// +z endcap
    MapE theBackwardLayers;  /// -z endcap 

    vector<MuonBarrelNavigableLayer*> theBarrelNLC;
    vector<MuonForwardNavigableLayer*> theForwardNLC;
    vector<MuonForwardNavigableLayer*> theBackwardNLC;

    const MuonDetLayerGeometry * theMuonDetLayerGeometry; 
  
};
#endif
