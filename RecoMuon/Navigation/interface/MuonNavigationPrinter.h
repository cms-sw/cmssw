#ifndef Navigation_MuonNavigationPrinter_H
#define Navigation_MuonNavigationPrinter_H

/** \class MuonNavigationPrinter
 *
 * Description:
 *  class to print the MuonNavigationSchool
 *
 * $Date: 2006/04/24 18:58:49 $
 * $Revision: 1.3 $
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

using namespace std;

class MuonNavigationPrinter {
  public:
    MuonNavigationPrinter(const MuonDetLayerGeometry *);
    MuonNavigationPrinter(const MuonDetLayerGeometry *,const GeometricSearchTracker *);

  private:
    void printLayer(DetLayer*) const;
    void printNextLayers(vector<const DetLayer*>) const;
    /// return detector part (barrel, forward, backward)
    string layerPart(const DetLayer*) const;
    /// return detector module (pixel, silicon, msgc, dt, csc, rpc)
    string layerModule(const DetLayer*) const;

};
#endif
