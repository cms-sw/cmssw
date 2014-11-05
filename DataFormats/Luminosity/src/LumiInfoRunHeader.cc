#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"

LumiInfoRunHeader::LumiInfoRunHeader(std::string& lumiProvider, std::string& fillingSchemeName,
				     std::bitset<LumiConstants::numBX>& fillingScheme, int bunchSpacing):
  lumiProvider_(lumiProvider),
  fillingSchemeName_(fillingSchemeName),
  fillingScheme_(fillingScheme),
  bunchSpacing_(bunchSpacing)
{
}

bool LumiInfoRunHeader::isProductEqual(LumiInfoRunHeader const& o) const
{
  return (lumiProvider_ == o.lumiProvider_ &&
	  fillingSchemeName_ == o.fillingSchemeName_ &&
	  fillingScheme_ == o.fillingScheme_);
}

//==============================================================================
