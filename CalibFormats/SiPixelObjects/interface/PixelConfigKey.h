#ifndef PixelConfigKey_h
#define PixelConfigKey_h
//
// This class implements the configuration key
// which actually just is an integer.
//

#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

namespace pos{
  class PixelConfigKey {

  public:

    explicit PixelConfigKey(unsigned int key) { key_=key;}

    unsigned int key() {return key_;}

  private:

    unsigned int key_;
    
  };
}
#endif
