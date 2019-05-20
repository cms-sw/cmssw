#ifndef PixelConfigurationVerifier_h
#define PixelConfigurationVerifier_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelConfigurationVerifier.h
*   \brief This class performs various tests to make sure that configurations are consistent
*
*   A longer explanation will be placed here later
*/
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "CalibFormats/SiPixelObjects/interface/PixelFEDCard.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"

namespace pos {

  /*! \class PixelConfigurationVerifier PixelConfigurationVerifier.h "interface/PixelConfigurationVerifier.h"
*   \brief This class performs various tests to make sure that configurations are consistent
*
*   A longer explanation will be placed here later
*/
  class PixelConfigurationVerifier {
  public:
    PixelConfigurationVerifier() {}
    virtual ~PixelConfigurationVerifier() {}

    //This method verifies that the right channels
    //are enabled on the set of FED card.
    //Warning messages are printed if a mismatch is found
    //and the fedcards are modified.
    void checkChannelEnable(PixelFEDCard *theFEDCard,
                            PixelNameTranslation *theNameTranslation,
                            PixelDetectorConfig *theDetConfig);

  private:
  };
}  // namespace pos
#endif
