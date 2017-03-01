#include "CondTools/RPC/interface/RPCLBLinkNameParser.h"

#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"

void RPCLBLinkNameParser::parse(std::string const & _name, RPCLBLink & _lb_link)
{
    _lb_link.reset();
    std::string::size_type _size = _name.size();
    std::string::size_type _pos(0), _next(0);
    int _tmp;

    std::istringstream _conv;

    // region
    _pos = _name.find("_R", _pos);
    if (_pos == std::string::npos || (_pos += 2) >= _size)
        throw cms::Exception("InvalidLinkBoardName") << "Expected _R[region], got " << _name;
    switch (_name.at(_pos)) {
    case 'B': _lb_link.setRegion(0); break;
    case 'E': _lb_link.setRegion(1); break;
    default:
        throw cms::Exception("InvalidLinkBoardName") << "Expected Region B or E, got " << _name.at(_pos) << " in " << _name;
        break;
    }
    if ((++_pos) >= _size)
        throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << _name;

    // yoke
    _next = _name.find_first_not_of("+-0123456789", _pos);
    _conv.clear();
    _conv.str(_name.substr(_pos, _next - _pos));
    _conv >> _tmp;
    _lb_link.setYoke(_tmp);
    _pos = _next;

    // sector
    _pos = _name.find("_S", _pos);
    if (_pos == std::string::npos || (_pos += 2) >= _size)
        throw cms::Exception("InvalidLinkBoardName") << "Expected _S[sector], got " << _name;
    _next = _name.find_first_not_of("+-0123456789", _pos);
    _conv.clear();
    _conv.str(_name.substr(_pos, _next - _pos));
    _conv >> _tmp;
    _lb_link.setSector(_tmp);
    _pos = _next;

    // (region) side
    _pos = _name.find("_", _pos);
    if (_pos == std::string::npos || (_pos += 2) >= _size)
        throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << _name;
    switch (_name.at(_pos)) {
    case 'N': _lb_link.setSide(0); break;
    case 'M': _lb_link.setSide(1); break;
    case 'P': _lb_link.setSide(2); break;
    default:
        throw cms::Exception("InvalidLinkBoardName") << "Expected Side N, M or P, got " << _name.at(_pos) << " in " << _name;
        break;
    }
    if ((++_pos) >= _size)
        throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << _name;

    // wheelordisk
    _conv.clear();
    _conv.str(_name.substr(_pos, 1));
    _conv >> _tmp;
    _lb_link.setWheelOrDisk(_tmp);
    if ((++_pos) >= _size)
        throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << _name;

    // fibre
    {
        std::string _fibre("123ABCDE");
        char const * _tmpchar = std::find(&(_fibre[0]), &(_fibre[0]) + 8, _name.at(_pos));
        _lb_link.setFibre(_tmpchar - &(_fibre[0]));
    }
    if ((++_pos) >= _size)
        return;

    // radial
    _next = _name.find("_CH", _pos);
    if (_next == std::string::npos)
        _next = _size;
    if (_next - _pos == 2) {
        std::string _radial = _name.substr(_pos, 2);
        if (_radial == "ab")
            _lb_link.setRadial(0);
        else if (_radial == "cd")
            _lb_link.setRadial(1);
    }

    if (_next == _size)
        return;

    // linkboard
    _pos = _next;
    if (_pos + 3 >= _size)
        throw cms::Exception("InvalidLinkBoardName") << "Name too short: " << _name;
    _pos += 3;
    _next = _name.find_first_not_of("+-0123456789", _pos);
    _conv.clear();
    _conv.str(_name.substr(_pos, _next - _pos));
    _conv >> _tmp;
    _lb_link.setLinkBoard(_tmp);
}


RPCLBLink RPCLBLinkNameParser::parse(std::string const & _name)
{
    RPCLBLink _lb_link;
    parse(_name, _lb_link);
    return _lb_link;
}
