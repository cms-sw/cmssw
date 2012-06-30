#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Alignment/CommonAlignment/interface/Counters.h"

using namespace align;

//__________________________________________________________________________________________________
Counter Counters::get(StructureType type) const
{
  auto n = theCounters.find(type);

  if (theCounters.end() == n)
    {
      throw cms::Exception("SetupError")
        << "Cannot find counter corresponding to the structure "
        << AlignableObjectId::idToString(type);
    }

  return n->second;
}
