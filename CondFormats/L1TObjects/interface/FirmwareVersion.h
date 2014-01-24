///
/// \class l1t::FirmwareVersion
///
/// Description: Common FirmwareVersion container for L1T subsystems
///
/// Implementation:
///    
///
/// \author: Michael Mulhearn - UC Davis
///

#ifndef L1T_FIRMWAREVERSION_H
#define L1T_FIRMWAREVERSION_H

#include <iostream>

namespace l1t {
  
  class FirmwareVersion {
    
  public:
    // Default constructor is required for CondFormats classes:
    FirmwareVersion() {}

    // Const getters for parameters:
    unsigned firmwareVersion() const {return m_fw_version;}
    
    // Non-const setters:
    void setFirmwareVersion(unsigned v) {m_fw_version=v;};

    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const FirmwareVersion & x) { x.print(o); return o; } 
    
  private:
    unsigned m_fw_version;
  };

}// namespace
#endif
