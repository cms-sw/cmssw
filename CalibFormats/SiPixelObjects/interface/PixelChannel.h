#ifndef PixelChannel_h
#define PixelChannel_h
 
#include <string>
#include <iostream>
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTBMChannel.h"

// class holding module name and TBM channel ("A" or "B") associated with a channel

namespace pos{
  class PixelChannel
  {
    public:

    PixelChannel(){;}
    PixelChannel(PixelModuleName module, std::string TBMChannel);
    PixelChannel(PixelModuleName module, PixelTBMChannel TBMChannel);

    const PixelModuleName& module() const { return module_; }
    std::string modulename() const { return module_.modulename(); }
    const PixelTBMChannel& TBMChannel() const { return TBMChannel_; }
    std::string TBMChannelString() const { return TBMChannel_.string(); }

    friend std::ostream& operator<<(std::ostream& s, const PixelChannel& channel);

    // allows for use of find() function in a map of PixelChannels
    const bool operator<(const PixelChannel& aChannel) const{
      return (module_<aChannel.module_ || (module_==aChannel.module_ && TBMChannel_ < aChannel.TBMChannel_) );
    }
    
    const bool operator==(const PixelChannel& aChannel) const{
      return (module_==aChannel.module_ && TBMChannel_==aChannel.TBMChannel_);
    }

    private:
    PixelModuleName module_    ;
    PixelTBMChannel TBMChannel_;
  };
}
#endif
