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
  s << channel.modulename() << "_" << channel.TBMChannelString();
  return s;
}
