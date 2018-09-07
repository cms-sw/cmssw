#ifndef Navigation_MTDNavigationPrinter_H
#define Navigation_MTDNavigationPrinter_H

/** \class MTDNavigationPrinter
 *
 * Description:
 *  class to print the MTDNavigationSchool
 *
 *
 * \author : L. Gray - FNAL
 *
 */

class DetLayer;
class MTDDetLayerGeometry;
class GeometricSearchTracker;
class MTDNavigationSchool;

#include <vector>
#include <string>

class MTDNavigationPrinter {
  public:

  MTDNavigationPrinter(const MTDDetLayerGeometry *, MTDNavigationSchool const &, bool enableBTL = true, bool enableETL = true );
  MTDNavigationPrinter(const MTDDetLayerGeometry *,MTDNavigationSchool const &, const GeometricSearchTracker *);

  private:
    void printLayer(const DetLayer*) const;
    void printLayers(const std::vector<const DetLayer*>&) const;
    /// return detector part (barrel, forward, backward)
//    std::string layerPart(const DetLayer*) const;
    /// return detector module (pixel, silicon, msgc, dt, csc, rpc)
//    std::string layerModule(const DetLayer*) const;


  MTDNavigationSchool const * school=nullptr;

};
#endif
