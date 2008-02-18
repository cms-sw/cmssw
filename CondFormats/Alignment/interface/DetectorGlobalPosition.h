#ifndef __CondFormats_Alignment_DetectorGlobalPosition_h
#define __CondFormats_Alignment_DetectorGlobalPosition_h

///
/// A function to extract GlobalPositionRcd from a given record and
/// return the entry corresponding to this detector id
///

#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace align {
  const AlignTransform &DetectorGlobalPosition(const edm::ESHandle<Alignments> &allGlobals, const DetId &id) {
    for (std::vector<AlignTransform>::const_iterator iter = allGlobals->m_align.begin();
	 iter != allGlobals->m_align.end();
	 ++iter) {
      if (iter->rawId() == id.rawId()) {
	return *iter;
      }
    }

    throw cms::Exception("RecordNotFound")
      << "DetId(" << id.rawId() << ") not found in GlobalPositionRcd" << std::endl;
  }
}

#endif
