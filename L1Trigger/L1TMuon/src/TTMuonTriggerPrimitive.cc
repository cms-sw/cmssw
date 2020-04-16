#include "L1Trigger/L1TMuon/interface/TTMuonTriggerPrimitive.h"

#include <iostream>

using namespace L1TMuon;

namespace {
  const char subsystem_names[][3] = {"TT"};
}

// Constructor from track trigger digi
TTTriggerPrimitive::TTTriggerPrimitive(const TTDetId& detid, const TTDigi& digi) : _id(detid), _subsystem(kTT) {
  calculateTTGlobalSector(detid, _globalsector, _subsector);

  const MeasurementPoint& mp = digi.clusterRef(0)->findAverageLocalCoordinatesCentered();
  _data.row_f = mp.x();
  _data.col_f = mp.y();
  _data.bend = digi.bendFE();
  _data.bx = 0;
}

// Copy constructor
TTTriggerPrimitive::TTTriggerPrimitive(const TTTriggerPrimitive& tp)
    : _data(tp._data),
      _id(tp._id),
      _subsystem(tp._subsystem),
      _globalsector(tp._globalsector),
      _subsector(tp._subsector),
      _eta(tp._eta),
      _phi(tp._phi),
      _rho(tp._rho),
      _theta(tp._theta) {}

// Assignment operator
TTTriggerPrimitive& TTTriggerPrimitive::operator=(const TTTriggerPrimitive& tp) {
  this->_data = tp._data;
  this->_id = tp._id;
  this->_subsystem = tp._subsystem;
  this->_globalsector = tp._globalsector;
  this->_subsector = tp._subsector;
  this->_eta = tp._eta;
  this->_phi = tp._phi;
  this->_rho = tp._rho;
  this->_theta = tp._theta;
  return *this;
}

// Equality operator
bool TTTriggerPrimitive::operator==(const TTTriggerPrimitive& tp) const {
  return (static_cast<int>(this->_data.row_f) == static_cast<int>(tp._data.row_f) &&
          static_cast<int>(this->_data.col_f) == static_cast<int>(tp._data.col_f) &&
          this->_data.bend == tp._data.bend && this->_data.bx == tp._data.bx && this->_id == tp._id &&
          this->_subsystem == tp._subsystem && this->_globalsector == tp._globalsector &&
          this->_subsector == tp._subsector);
}

const int TTTriggerPrimitive::getBX() const {
  switch (_subsystem) {
    case kTT:
      return _data.bx;
    default:
      throw cms::Exception("Invalid Subsytem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return -1;
}

const int TTTriggerPrimitive::getStrip() const {
  switch (_subsystem) {
    case kTT:
      return static_cast<int>(_data.row_f);
    default:
      throw cms::Exception("Invalid Subsytem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return -1;
}

const int TTTriggerPrimitive::getSegment() const {
  switch (_subsystem) {
    case kTT:
      return static_cast<int>(_data.col_f);
    default:
      throw cms::Exception("Invalid Subsytem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return -1;
}

const int TTTriggerPrimitive::getBend() const {
  switch (_subsystem) {
    case kTT:
      return static_cast<int>(_data.bend);
    default:
      throw cms::Exception("Invalid Subsytem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return -1;
}

void TTTriggerPrimitive::calculateTTGlobalSector(const TTDetId& detid, unsigned& globalsector, unsigned& subsector) {
  globalsector = 0;
  subsector = 0;
}

std::ostream& operator<<(std::ostream& os, const TTTriggerPrimitive::TTDetId& detid) {
  // Note that there is no endl to end the output
  os << " undefined";
  return os;
}

void TTTriggerPrimitive::print(std::ostream& out) const {
  unsigned idx = (unsigned)_subsystem;
  out << subsystem_names[idx] << " Trigger Primitive" << std::endl;
  out << "eta: " << _eta << " phi: " << _phi << " rho: " << _rho << " bend: " << _theta << std::endl;
  switch (_subsystem) {
    case kTT:
      out << detId() << std::endl;
      out << "Strip         : " << static_cast<int>(_data.row_f) << std::endl;
      out << "Segment       : " << static_cast<int>(_data.col_f) << std::endl;
      out << "Bend          : " << _data.bend << std::endl;
      out << "BX            : " << _data.bx << std::endl;
      break;
    default:
      throw cms::Exception("Invalid Subsytem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
}
