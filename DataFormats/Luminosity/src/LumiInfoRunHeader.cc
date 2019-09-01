#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"

LumiInfoRunHeader::LumiInfoRunHeader(std::string& lumiProvider,
                                     std::string& fillingSchemeName,
                                     std::bitset<LumiConstants::numBX>& fillingScheme)
    : lumiProvider_(lumiProvider), fillingSchemeName_(fillingSchemeName), fillingScheme_(fillingScheme) {
  setBunchSpacing();
}

bool LumiInfoRunHeader::isProductEqual(LumiInfoRunHeader const& o) const {
  return (lumiProvider_ == o.lumiProvider_ && fillingSchemeName_ == o.fillingSchemeName_ &&
          fillingScheme_ == o.fillingScheme_);
}

//==============================================================================

void LumiInfoRunHeader::setFillingScheme(const std::bitset<LumiConstants::numBX>& fillingScheme) {
  fillingScheme_ = fillingScheme;
  setBunchSpacing();
}

// This function determines the bunch spacing from the filling scheme
// and sets bunchSpacing_ accordingly.

void LumiInfoRunHeader::setBunchSpacing(void) {
  int lastFilledBunch = -1;
  int minimumSpacingFound = LumiConstants::numBX;

  for (unsigned int i = 0; i < LumiConstants::numBX; i++) {
    if (fillingScheme_[i]) {
      if (lastFilledBunch >= 0) {
        int thisSpacing = i - lastFilledBunch;
        if (thisSpacing < minimumSpacingFound)
          minimumSpacingFound = thisSpacing;
      }
      lastFilledBunch = i;
    }
  }

  // If no bunches are filled, then just leave bunchSpacing at 0
  if (lastFilledBunch == -1)
    bunchSpacing_ = 0;
  else
    bunchSpacing_ = LumiConstants::bxSpacingInt * minimumSpacingFound;
}
