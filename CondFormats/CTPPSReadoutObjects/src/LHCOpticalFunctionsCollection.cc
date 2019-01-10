// Original Author:  Jan Kašpar

#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsCollection.h"

#include "FWCore/Utilities/interface/Exception.h"

//----------------------------------------------------------------------------------------------------

void LHCOpticalFunctionsCollection::interpolateFunctions(double xangle, mapType &output) const
{
  for (const auto &p1 : m_functions1)
  {
    const auto it2 = m_functions2.find(p1.first);
    if (it2 == m_functions2.end())
      throw cms::Exception("LHCOpticalFunctionsCollection") << "Mismatch between m_functions1 and m_functions2.";

    LHCOpticalFunctionsSet *fs = LHCOpticalFunctionsSet::interpolate(m_xangle1, p1.second, m_xangle2, it2->second, xangle);

    output.emplace(p1.first, *fs);
  }
}
