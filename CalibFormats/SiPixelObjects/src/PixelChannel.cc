#include "CalibFormats/SiPixelObjects/interface/PixelChannel.h"

using namespace pos;

PixelChannel::PixelChannel(PixelModuleName module, std::string TBMChannel):
  module_(module), TBMChannel_(TBMChannel)
{}

PixelChannel::PixelChannel(PixelModuleName module, PixelTBMChannel TBMChannel):
  module_(module), TBMChannel_(TBMChannel)
{}

std::ostream& pos::operator<<(std::ostream& s, const PixelChannel& channel)
{
  s << channel.channelname();
  return s;
}

std::string PixelChannel::channelname() const
{
	return modulename() + "_ch" + TBMChannelString();
}
