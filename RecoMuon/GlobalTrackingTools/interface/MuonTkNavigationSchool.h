#ifndef RecoMuon_GlobalTrackingTools_MuonTkNavigationSchool_H
#define RecoMuon_GlobalTrackingTools_MuonTkNavigationSchool_H

/** \class MuonTkNavigationSchool
 *
 *  Navigation School for both the Muon system and
 *  the Tracker. 
 *
 *
 *  $Date: 2007/09/25 19:31:36 $
 *  $Revision: 1.1 $
 *
 * \author Chang Liu - Purdue University
 * \author Stefano Lacaprara - INFN Padova 
 */

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoMuon/Navigation/interface/MuonDetLayerMap.h"
#include <vector>

class BarrelDetLayer;
class ForwardDetLayer;
class NavigableLayer;
class SimpleBarrelNavigableLayer;
class SimpleForwardNavigableLayer;
class MuonBarrelNavigableLayer;
class MuonForwardNavigableLayer;
class GeometricSearchTracker;
class MuonDetLayerGeometry;
class MagneticField;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class MuonTkNavigationSchool : public NavigationSchool {

  public:

    /// constructor
    MuonTkNavigationSchool(const MuonDetLayerGeometry*, 
                           const GeometricSearchTracker*, 
                           const MagneticField*);

    /// destructor
    ~MuonTkNavigationSchool();

    /// return a vector of NavigableLayer*, from base class
    virtual std::vector<NavigableLayer*> navigableLayers() const;

  private:

    /// add barrel layer
    void addBarrelLayer(BarrelDetLayer*);

    /// add endcap layer (backward and forward)
    void addEndcapLayer(ForwardDetLayer*);

    /// link barrel layers
    void linkBarrelLayers();

    /// link endcap layers
    void linkEndcapLayers(const MapE&, 
                          std::vector<MuonForwardNavigableLayer*>&,
                          std::vector<SimpleForwardNavigableLayer*>&);

    /// calaulate the length of the barrel
    float barrelLength() const;

    /// pseudorapidity from r and z
    float calculateEta(float r, float z) const;

  private:

    struct delete_layer {
      template <typename T>
      void operator()(T*& p) {
        if (p) {
          delete p;
          p = 0;
        }
      }
    };

    typedef std::vector<BarrelDetLayer*>   BDLC;
    typedef std::vector<ForwardDetLayer*>  FDLC;

    MapB theBarrelLayers;
    MapE theForwardLayers;   // +z
    MapE theBackwardLayers;  // -z

    std::vector<SimpleBarrelNavigableLayer*>  theTkBarrelNLC;
    std::vector<SimpleForwardNavigableLayer*> theTkForwardNLC;
    std::vector<SimpleForwardNavigableLayer*> theTkBackwardNLC;

    std::vector<MuonBarrelNavigableLayer*>  theMuonBarrelNLC;
    std::vector<MuonForwardNavigableLayer*> theMuonForwardNLC;
    std::vector<MuonForwardNavigableLayer*> theMuonBackwardNLC;

    const MuonDetLayerGeometry* theMuonDetLayerGeometry;
    const GeometricSearchTracker* theGeometricSearchTracker;
    const MagneticField* theMagneticField;

};

#endif
