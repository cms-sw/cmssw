#include "CondFormats/RPCObjects/interface/RPCLBLink.h"

#include <ostream>
#include <sstream>

RPCLBLink::RPCLBLink()
    : id_(0x0)
{}

RPCLBLink::RPCLBLink(std::uint32_t const & _id)
    : id_(_id)
{}

RPCLBLink::RPCLBLink(int _region
                     , int _yoke
                     , int _sector
                     , int _side
                     , int _wheelordisk
                     , int _fibre
                     , int _radial
                     , int _linkboard
                     , int _connector)
    : id_(0x0)
{
    setRegion(_region);
    setYoke(_yoke);
    setSector(_sector);
    setSide(_side);
    setWheelOrDisk(_wheelordisk);
    setFibre(_fibre);
    setRadial(_radial);
    setLinkBoard(_linkboard);
    setConnector(_connector);
}

std::uint32_t RPCLBLink::getMask() const
{
    std::uint32_t _mask(0x0);
    if (id_ & mask_region_)
        _mask |= mask_region_;
    if (id_ & mask_yoke_)
        _mask |= mask_yoke_;
    if (id_ & mask_sector_)
        _mask |= mask_sector_;
    if (id_ & mask_side_)
        _mask |= mask_side_;
    if (id_ & mask_wheelordisk_)
        _mask |= mask_wheelordisk_;
    if (id_ & mask_fibre_)
        _mask |= mask_fibre_;
    if (id_ & mask_radial_)
        _mask |= mask_radial_;
    if (id_ & mask_linkboard_)
        _mask |= mask_linkboard_;
    if (id_ & mask_connector_)
        _mask |= mask_connector_;
    return _mask;
}

std::string RPCLBLink::getName() const
{
    // LB_Rregion.yoke_Ssector_region.side.wheel_or_disk.fibre.radial_CHlinkboard:connector
    // LB_RB      -2  _S10    _ B     N    2             A           _CH2        :
    // LB_RE      -1  _S10    _ E     N    2             3           _CH0        :
    // LB_RB-2_S10_BN2A_CH2 , RB1in/W-2/S10:bwd ; LB_RE-1_S10_EN23_CH0 , RE-2/R3/C30

    int _region(getRegion())
        , _yoke(getYoke())
        , _linkboard(getLinkBoard())
        , _connector(getConnector());

    std::ostringstream _oss;
    _oss << "LB_R";
    switch (_region) {
    case 0:  _oss << 'B'; break;
    case 1:  _oss << 'E'; break;
    default: _oss << '*'; break;
    }
    (_yoke > 0 ? _oss << '+' << _yoke : _oss << _yoke);

    bf_stream(_oss << "_S", min_sector_, mask_sector_, pos_sector_);

    _oss << '_';
    switch (_region) {
    case 0:  _oss << 'B'; break;
    case 1:  _oss << 'E'; break;
    default: _oss << '*'; break;
    }
    switch (getSide()) {
    case 0:  _oss << 'N'; break;
    case 1:  _oss << 'M'; break;
    case 2:  _oss << 'P'; break;
    default: _oss << '*'; break;
    }
    bf_stream(_oss, min_wheelordisk_, mask_wheelordisk_, pos_wheelordisk_);
    switch (getFibre()) {
    case 0:  _oss << '1'; break;
    case 1:  _oss << '2'; break;
    case 2:  _oss << '3'; break;
    case 3:  _oss << 'A'; break;
    case 4:  _oss << 'B'; break;
    case 5:  _oss << 'C'; break;
    case 6:  _oss << 'D'; break;
    case 7:  _oss << 'E'; break;
    default: _oss << '*'; break;
    }
    switch (getRadial()) { // for completeness, CMS IN 2002/065
    case 0:  _oss << "ab"; break;
    case 1:  _oss << "cd"; break;
    default: _oss << "";   break;
    }

    if (_linkboard != wildcard_)
        _oss << "_CH" << _linkboard;

    if (_connector != wildcard_)
        _oss << ":" << _connector;

    return _oss.str();
}

std::ostream & operator<<(std::ostream & _ostream, RPCLBLink const & _link)
{
    return (_ostream << _link.getName());
}
