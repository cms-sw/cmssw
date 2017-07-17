#include "Alignment/CommonAlignment/interface/AlignableIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace align;

//__________________________________________________________________________________________________
Counter AlignableIndexer::get(StructureType type,
                              const AlignableObjectId& alignableObjectId) const
{
  auto n = theCounters.find(type);

  if (theCounters.end() == n)
    {
      throw cms::Exception("AlignableBuildProcess")
        << "Cannot find counter corresponding to the structure "
        << alignableObjectId.idToString(type);
    }

  return n->second;
}
