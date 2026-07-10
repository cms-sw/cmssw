#ifndef __CondFormats_Alignment_DetectorGlobalPosition_h
#define __CondFormats_Alignment_DetectorGlobalPosition_h

///
/// A function to extract GlobalPositionRcd from a given record and
/// return the entry corresponding to this detector id
///

#include "DataFormats/DetId/interface/DetIdFwd.h"

class Alignments;
class AlignTransform;

namespace align {
  const AlignTransform &DetectorGlobalPosition(const Alignments &allGlobals, const DetId &id);
}

#endif
