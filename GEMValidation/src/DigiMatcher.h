#ifndef _DigiMatcher_h_
#define _DigiMatcher_h_

/**\class DigiMatcher

 Description: Base class for matching of CSC or GEM Digis to SimTrack

 Original Author:  "Vadim Khotilovich"
 $Id$
*/

#include "BaseMatcher.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <vector>
#include <tuple>

class SimHitMatcher;
class CSCGeometry;
class GEMGeometry;
class CSCLayerGeometry;

class DigiMatcher : public BaseMatcher
{
public:
  
  typedef enum {GEM_STRIP=0, GEM_PAD, GEM_COPAD, CSC_STRIP, CSC_WIRE} DigiType;
  // digi info keeper: <detid, channel, bx, type>
  typedef std::tuple<unsigned int, int, int, DigiType> Digi;
  typedef std::vector<Digi> DigiContainer;

  DigiMatcher(SimHitMatcher& sh);
  
  ~DigiMatcher();

  /// calculate Global position for a digi
  /// works for GEM and CSC strip digis
  GlobalPoint digiPosition(const Digi& digi);

  /// calculate Global average position for a provided collection of digis
  /// works for GEM and CSC strip digis
  GlobalPoint digisMeanPosition(const DigiContainer& digis);

  /// for CSC strip and wire:
  /// first calculate median half-strip and widegroup
  /// then use CSCLayerGeometry::intersectionOfStripAndWire to calculate the intersection
  GlobalPoint digisCSCMedianPosition(const DigiContainer& strip_digis, const DigiContainer& wire_digis);

  /// calculate median strip (or wiregroup for wire digis) in a set
  /// assume that the set of digis was from layers of a single chamber
  int median(const DigiContainer& digis);

  /// for GEM:
  /// find a GEM digi with its position that is the closest in deltaR to the provided CSC global position
  std::pair<DigiMatcher::Digi, GlobalPoint>
  digiInGEMClosestToCSC(const DigiContainer& gem_digis, const GlobalPoint& csc_gp);

protected:

  SimHitMatcher* simhit_matcher_;

  const CSCGeometry* csc_geo_;
  const GEMGeometry* gem_geo_;
};

#endif
