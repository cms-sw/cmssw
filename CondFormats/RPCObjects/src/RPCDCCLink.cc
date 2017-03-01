#include "CondFormats/RPCObjects/interface/RPCDCCLink.h"

#include <ostream>
#include <sstream>

RPCDCCLink::RPCDCCLink()
    : id_(0x0)
{}

RPCDCCLink::RPCDCCLink(::uint32_t const & _id)
    : id_(_id)
{}

RPCDCCLink::RPCDCCLink(int _fed
                       , int _dccinput
                       , int _tbinput)
    : id_(0x0)
{
    setFED(_fed);
    setDCCInput(_dccinput);
    setTBInput(_tbinput);
}

::uint32_t RPCDCCLink::getMask() const
{
    ::uint32_t _mask(0x0);
    if (id_ & mask_fed_)
        _mask |= mask_fed_;
    if (id_ & mask_dccinput_)
        _mask |= mask_dccinput_;
    if (id_ & mask_tbinput_)
        _mask |= mask_tbinput_;
    return _mask;
}

std::string RPCDCCLink::getName() const
{
    std::ostringstream _oss;
    _oss << "RPCDCCLink_";
    bf_stream(_oss, min_fed_, mask_fed_, pos_fed_);
    bf_stream(_oss << '_', min_dccinput_, mask_dccinput_, pos_dccinput_);
    bf_stream(_oss << '_', min_tbinput_, mask_tbinput_, pos_tbinput_);
    return _oss.str();
}

std::ostream & operator<<(std::ostream & _ostream, RPCDCCLink const & _link)
{
    return (_ostream << _link.getName());
}
