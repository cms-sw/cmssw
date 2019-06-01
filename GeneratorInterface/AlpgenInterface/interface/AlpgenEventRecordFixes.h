#ifndef GeneratorInterface_AlpgenInterface_AlpgenEventRecordFixes_h
#define GeneratorInterface_AlpgenInterface_AlpgenEventRecordFixes_h

#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

namespace alpgen {
  /// Functions to fixthe HEPEUP Event Record, adding the particles that
  /// ALPGEN skips in the .unw event.

  /// A function to return a LorentzVector from a given position
  /// in the HEPEUP
  math::XYZTLorentzVector vectorFromHepeup(const lhef::HEPEUP &hepeup, int index);

  /// Fixes Event Record for ihrd = 1,2,3,4,10,14,15
  void fixEventWZ(lhef::HEPEUP &hepeup);

  /// Fixes Event Record for ihrd = 5
  void fixEventMultiBoson(lhef::HEPEUP &hepeup);

  /// Fixes Event Record for ihrd = 6
  void fixEventTTbar(lhef::HEPEUP &hepeup);

  /// Fixes Event Record for ihrd = 8
  void fixEventHiggsTTbar(lhef::HEPEUP &hepeup);

  /// Fixes Event Record for ihrd = 13
  void fixEventSingleTop(lhef::HEPEUP &hepeup, double mb, int itopprc);

}  // namespace alpgen

#endif  // GeneratorInterface_AlpgenInterface_AlpgenEventRecordFixes_h
