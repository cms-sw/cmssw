#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"

LumiInfoRunHeader::LumiInfoRunHeader(std::string& lumiProvider, std::string& fillingSchemeName,
				     vbool_t& fillingScheme):
  m_lumiProvider(lumiProvider),
  m_fillingSchemeName(fillingSchemeName)
{
  m_fillingScheme.assign(fillingScheme.begin(), fillingScheme.end());
}

bool LumiInfoRunHeader::isProductEqual(LumiInfoRunHeader const& o) const
{
  return (m_lumiProvider == o.m_lumiProvider &&
	  m_fillingSchemeName == o.m_fillingSchemeName &&
	  m_fillingScheme == o.m_fillingScheme);
}

//==============================================================================

void LumiInfoRunHeader::setFillingScheme(const vbool_t& fillingScheme)
{
  m_fillingScheme.assign(fillingScheme.begin(), fillingScheme.end());
}
