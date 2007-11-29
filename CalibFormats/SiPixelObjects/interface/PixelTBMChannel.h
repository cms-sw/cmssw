#ifndef PixelTBMChannel_h
#define PixelTBMChannel_h
 
#include <string>
#include <iostream>

// simple class to hold either "A" or "B" for the TBM channel

namespace pos{
  class PixelTBMChannel
  {
    public:

    PixelTBMChannel(std::string TBMChannel);

    std::string string() const;

    friend std::ostream& operator<<(std::ostream& s, const PixelTBMChannel& TBMChannel);

    const bool operator<(const PixelTBMChannel& aTBMChannel) const{
      return ( isChannelB_ == false && aTBMChannel.isChannelB_ == true );
    }

    private:
    bool isChannelB_;
  };
}
#endif
