#ifndef Navigation_MuonNavigationPrinter_H
#define Navigation_MuonNavigationPrinter_H

/** \class MuonNavigationPrinter
 *
 * Description:
 *  class to print the MuonNavigationSchool
 *
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * The class prints nextLayers and compatibleLayers
 * Add new constructor for MuonTkNavigationSchool
 */

class DetLayer;
class MuonDetLayerGeometry;
class GeometricSearchTracker;

#include <vector>
#include <string>

class MuonNavigationPrinter {
  public:
    MuonNavigationPrinter(const MuonDetLayerGeometry *, bool enableRPC = true );
    MuonNavigationPrinter(const MuonDetLayerGeometry *,const GeometricSearchTracker *);

  private:
    void printLayer(DetLayer*) const;
    void printLayers(const std::vector<const DetLayer*>&) const;
    /// return detector part (barrel, forward, backward)
//    std::string layerPart(const DetLayer*) const;
    /// return detector module (pixel, silicon, msgc, dt, csc, rpc)
//    std::string layerModule(const DetLayer*) const;

};
#endif
