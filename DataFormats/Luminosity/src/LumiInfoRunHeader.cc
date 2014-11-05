#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"

LumiInfoRunHeader::LumiInfoRunHeader(std::string& lumiProvider, std::string& fillingSchemeName,
				     std::bitset<LumiConstants::numBX>& fillingScheme):
  m_lumiProvider(lumiProvider),
  m_fillingSchemeName(fillingSchemeName)
{
  m_fillingScheme = fillingScheme;
}

bool LumiInfoRunHeader::isProductEqual(LumiInfoRunHeader const& o) const
{
  return (m_lumiProvider == o.m_lumiProvider &&
	  m_fillingSchemeName == o.m_fillingSchemeName &&
	  m_fillingScheme == o.m_fillingScheme);
}

//==============================================================================

void LumiInfoRunHeader::setFillingScheme(const std::bitset<LumiConstants::numBX>& fillingScheme)
{
  m_fillingScheme = fillingScheme;
}
