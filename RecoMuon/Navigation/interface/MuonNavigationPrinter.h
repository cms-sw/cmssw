#ifndef Navigation_MuonNavigationPrinter_H
#define Navigation_MuonNavigationPrinter_H

/** \class MuonNavigationPrinter
 *
 * Description:
 *  class to print the MuonNavigationSchool
 *
 * $Date: $
 * $Revision: $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * The class prints nextLayers and compatibleLayers
 */

class DetLayer;
class MuonDetLayerGeometry;

#include <vector>
#include <string>

using namespace std;

class MuonNavigationPrinter {
  public:
    MuonNavigationPrinter(const MuonDetLayerGeometry *);
  private:
    void printLayer(DetLayer*) const;
    void printNextLayers(vector<const DetLayer*>) const;
    /// return detector part (barrel, forward, backward)
    string layerPart(const DetLayer*) const;
    /// return detector module (pixel, silicon, msgc, dt, csc, rpc)
    string layerModule(const DetLayer*) const;

};
#endif
