#ifndef Navigation_MuonNavigationPrinter_H
#define Navigation_MuonNavigationPrinter_H

// Ported from ORCA
// MuonNavigationPrinter prints out MuonNavigationSchool
//   $Date: $
//   $Revision: $

class DetLayer;

#include <vector>
#include <string>

using namespace std;

class MuonNavigationPrinter {
  public:
    MuonNavigationPrinter();
  private:
    void printLayer(DetLayer*) const;
    void printNextLayers(vector<const DetLayer*>) const;
    /// return detector part (barrel, forward, backward)
    string layerPart(const DetLayer*) const;
};
#endif
