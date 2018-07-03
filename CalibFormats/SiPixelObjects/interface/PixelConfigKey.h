#ifndef PixelConfigKey_h
#define PixelConfigKey_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelConfigKey.h
*   \brief This class implements the configuration key which actually just is an integer.
*
*   A longer explanation will be placed here later
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

namespace pos{
/*! \class PixelConfigKey PixelConfigKey.h "interface/PixelConfigKey.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelConfigKey {

  public:

    explicit PixelConfigKey(unsigned int key) { key_=key;}
    unsigned int key() {return key_;}

  private:

    unsigned int key_;
    
  };
}
#endif
