#ifndef PixelTBMChannel_h
#define PixelTBMChannel_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelTBMChannel.h
*   \brief Simple class to hold either "A" or "B" for the TBM channel
*
*    A longer explanation will be placed here later
*/
 
#include <string>
#include <iostream>

namespace pos{
/*! \class PixelTBMChannel PixelTBMChannel.h "interface/PixelTBMChannel.h"
*   \brief Simple class to hold either "A" or "B" for the TBM channel
*
*   A longer explanation will be placed here later
*/
  class PixelTBMChannel
  {
    public:

    PixelTBMChannel(){;}
    PixelTBMChannel(std::string TBMChannel);

    std::string string() const;

    friend std::ostream& operator<<(std::ostream& s, const PixelTBMChannel& TBMChannel);

    const bool operator<(const PixelTBMChannel& aTBMChannel) const{
      return ( isChannelB_ == false && aTBMChannel.isChannelB_ == true );
    }

    const bool operator==(const PixelTBMChannel& aTBMChannel) const{
      return isChannelB_==aTBMChannel.isChannelB_;
    }

    private:
    bool isChannelB_;
  };
  std::ostream& operator<<(std::ostream& s, const PixelTBMChannel& TBMChannel);
}
#endif
