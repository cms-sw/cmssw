#ifndef GEMValidation_DigiMatcher_h
#define GEMValidation_DigiMatcher_h

/**\class DigiMatcher

 Description: Base class for matching of CSC or GEM Digis to SimTrack

 Original Author:  "Vadim Khotilovich"
*/

#include "GEMCode/GEMValidation/src/BaseMatcher.h"
#include "GEMCode/GEMValidation/src/GenericDigi.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class SimHitMatcher;

class DigiMatcher : public BaseMatcher
{
public:

  typedef matching::Digi Digi;
  typedef matching::DigiContainer DigiContainer;

  DigiMatcher(SimHitMatcher& sh);
  
  ~DigiMatcher();

  /// calculate Global position for a digi
  /// works for GEM and CSC strip digis
  GlobalPoint digiPosition(const Digi& digi) const;

  /// calculate Global average position for a provided collection of digis
  /// works for GEM and CSC strip digis
  GlobalPoint digisMeanPosition(const DigiContainer& digis) const;

  /// for CSC strip and wire:
  /// first calculate median half-strip and widegroup
  /// then use CSCLayerGeometry::intersectionOfStripAndWire to calculate the intersection
  GlobalPoint digisCSCMedianPosition(const DigiContainer& strip_digis, const DigiContainer& wire_digis) const;

  /// calculate median strip (or wiregroup for wire digis) in a set
  /// assume that the set of digis was from layers of a single chamber
  int median(const DigiContainer& digis) const;

  /// for GEM:
  /// find a GEM digi with its position that is the closest in deltaR to the provided CSC global position
  std::pair<Digi, GlobalPoint>
  digiInGEMClosestToCSC(const DigiContainer& gem_digis, const GlobalPoint& csc_gp) const;

protected:

  const SimHitMatcher* simhit_matcher_;

  const DigiContainer no_digis_;
};

#endif
