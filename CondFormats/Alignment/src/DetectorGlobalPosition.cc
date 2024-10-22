#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>

namespace align {
  const AlignTransform &DetectorGlobalPosition(const Alignments &allGlobals, const DetId &id) {
    for (std::vector<AlignTransform>::const_iterator iter = allGlobals.m_align.begin();
         iter != allGlobals.m_align.end();
         ++iter) {
      if (iter->rawId() == id.rawId()) {
        return *iter;
      }
    }

    throw cms::Exception("RecordNotFound") << "DetId(" << id.rawId() << ") not found in GlobalPositionRcd" << std::endl;
  }
}  // namespace align
