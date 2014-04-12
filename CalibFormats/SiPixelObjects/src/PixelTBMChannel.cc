#include "CalibFormats/SiPixelObjects/interface/PixelTBMChannel.h"
#include <cassert>

using namespace pos;

PixelTBMChannel::PixelTBMChannel(std::string TBMChannel)
{
	if      ( TBMChannel=="A" ) isChannelB_ = false;
	else if ( TBMChannel=="B" ) isChannelB_ = true;
	else
	{
		std::cout << "ERROR in PixelTBMChannel: TBM channel must be A or B, but input value was "<<TBMChannel<<std::endl;
		assert(0);
	}
}

std::string PixelTBMChannel::string() const
{
	if ( isChannelB_ ) return "B";
	else              return "A";
}

std::ostream& pos::operator<<(std::ostream& s, const PixelTBMChannel& TBMChannel)
{
	s << TBMChannel.string();
	return s;
}
