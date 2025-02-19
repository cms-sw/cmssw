#ifndef __CondFormats_Alignment_DetectorGlobalPosition_h
#define __CondFormats_Alignment_DetectorGlobalPosition_h

///
/// A function to extract GlobalPositionRcd from a given record and
/// return the entry corresponding to this detector id
///

class Alignments;
class AlignTransform;
class DetId;

namespace align {
  const AlignTransform &DetectorGlobalPosition(const Alignments &allGlobals, const DetId &id);
}

#endif
