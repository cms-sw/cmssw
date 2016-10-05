#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"

#include <ostream>
#include <sstream>

RPCAMCLink::RPCAMCLink()
    : id_(0x0)
{}

RPCAMCLink::RPCAMCLink(::uint32_t const & _id)
    : id_(_id)
{}

RPCAMCLink::RPCAMCLink(int _fed
                       , int _amcnumber
                       , int _amcinput)
    : id_(0x0)
{
    setFED(_fed);
    setAMCNumber(_amcnumber);
    setAMCInput(_amcinput);
}

::uint32_t RPCAMCLink::getMask() const
{
    ::uint32_t _mask(0x0);
    if (id_ & mask_fed_)
        _mask |= mask_fed_;
    if (id_ & mask_amcnumber_)
        _mask |= mask_amcnumber_;
    if (id_ & mask_amcinput_)
        _mask |= mask_amcinput_;
    return _mask;
}

std::string RPCAMCLink::getName() const
{
    std::ostringstream _oss;
    _oss << "RPCAMCLink_";
    bf_stream(_oss, min_fed_, mask_fed_, pos_fed_);
    bf_stream(_oss << '_', min_amcnumber_, mask_amcnumber_, pos_amcnumber_);
    bf_stream(_oss << '_', min_amcinput_, mask_amcinput_, pos_amcinput_);
    return _oss.str();
}

std::ostream & operator<<(std::ostream & _ostream, RPCAMCLink const & _link)
{
    return (_ostream << _link.getName());
}
