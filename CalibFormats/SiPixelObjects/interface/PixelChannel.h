#ifndef PixelChannel_h
#define PixelChannel_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelChannel.h
*   \brief This class implements...
*
*   A longer explanation will be placed here later
*/
 
#include <string>
#include <iostream>
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTBMChannel.h"

// class holding module name and TBM channel ("A" or "B") associated with a channel

namespace pos{
/*! \class PixelChannel PixelChannel.h "interface/PixelChannel.h"
*
*   A longer explanation will be placed here later
*/
  class PixelChannel
  {
    public:

    PixelChannel(){;}
    PixelChannel(PixelModuleName module, std::string TBMChannel);
    PixelChannel(PixelModuleName module, PixelTBMChannel TBMChannel);
    PixelChannel(std::string name); // takes a name of the form produced by channelname()

    const PixelModuleName& module() const { return module_; }
    std::string modulename() const { return module_.modulename(); }
    const PixelTBMChannel& TBMChannel() const { return TBMChannel_; }
    std::string TBMChannelString() const { return TBMChannel_.string(); }

    std::string channelname() const;

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

  std::ostream& operator<<(std::ostream& s, const PixelChannel& channel);
}

#endif
