#ifndef PixelConfigurationVerifier_h
#define PixelConfigurationVerifier_h
//
// This class performs various tests to make
// sure that configurations are consistent
// 
// 
// 
//
//
//
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "CalibFormats/SiPixelObjects/interface/PixelFEDCard.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"

namespace pos{

  class PixelConfigurationVerifier {

  public:

    PixelConfigurationVerifier(){}
    virtual ~PixelConfigurationVerifier(){}
    
    //This method verifies that the right channels
    //are enabled on the set of FED card.
    //Warning messages are printed if a mismatch is found
    //and the fedcards are modified.
    void checkChannelEnable(PixelFEDCard *theFEDCard,
			    PixelNameTranslation *theNameTranslation,
			    PixelDetectorConfig *theDetConfig);
 
  private:

  };
}
#endif
