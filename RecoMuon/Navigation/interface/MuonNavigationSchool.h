#ifndef Navigation_MuonNavigationSchool_H
#define Navigation_MuonNavigationSchool_H

//   Ported from ORCA.
//   The MuonNavigationSchool will provide two kinds of maps. 
//   One is for nextLayers(), one for compatibleLayers().
//   Both are implemented in MuonBarrelNavigableLayer and MuonForwardNavigableLayer.
//   $Date: $
//   $Revision: $

#include "RecoMuon/Navigation/interface/MuonLayerSort.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
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
    MuonNavigationSchool();
    // Destructor
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
  
    MapB theBarrelLayers;    // barrel
    MapE theForwardLayers;   // +z endcap
    MapE theBackwardLayers;  // -z endcap 

    vector<MuonBarrelNavigableLayer*> theBarrelNLC;
    vector<MuonForwardNavigableLayer*> theForwardNLC;
    vector<MuonForwardNavigableLayer*> theBackwardNLC;

};
#endif
