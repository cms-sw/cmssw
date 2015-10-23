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
class MuonNavigationSchool;

#include <vector>
#include <string>

class MuonNavigationPrinter {
  public:
  MuonNavigationPrinter(const MuonDetLayerGeometry *, MuonNavigationSchool const &, bool enableRPC = true,
                        bool enableCSC = true, bool enableGEM = false  );
  MuonNavigationPrinter(const MuonDetLayerGeometry *,MuonNavigationSchool const &, const GeometricSearchTracker *);

  private:
    void printLayer(const DetLayer*) const;
    void printLayers(const std::vector<const DetLayer*>&) const;
    /// return detector part (barrel, forward, backward)
//    std::string layerPart(const DetLayer*) const;
    /// return detector module (pixel, silicon, msgc, dt, csc, rpc)
//    std::string layerModule(const DetLayer*) const;


  MuonNavigationSchool const * school=nullptr;

};
#endif
