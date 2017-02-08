#include "Alignment/CommonAlignment/interface/AlignableIndexer.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace align;

//__________________________________________________________________________________________________
Counter AlignableIndexer::get(StructureType type) const
{
  auto n = theCounters.find(type);

  if (theCounters.end() == n)
    {
      throw cms::Exception("AlignableBuildProcess")
        << "Cannot find counter corresponding to the structure "
        << AlignableObjectId::idToString(type);
    }

  return n->second;
}
