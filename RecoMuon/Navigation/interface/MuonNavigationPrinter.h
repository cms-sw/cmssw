#ifndef Navigation_MuonNavigationPrinter_H
#define Navigation_MuonNavigationPrinter_H

// Ported from ORCA
// MuonNavigationPrinter prints out MuonNavigationSchool
//   $Date: 2006/03/22 01:54:25 $
//   $Revision: 1.1 $

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
