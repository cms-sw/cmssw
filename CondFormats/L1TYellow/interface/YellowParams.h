///
/// \class l1t::YellowParams
///
/// Description: Configuration parameters for the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to implement CondFormat class.
///
/// \author: Michael Mulhearn - UC Davis
///

#ifndef YELLOWPARAMS_H
#define YELLOWPARAMS_H

//
//  This class demonstrates how to store configuration data for the emulator.
//
//  As firmware evolves, the types and number of parameters may change.  This
//  interface should be extended (but not diminished) to include any new
//  parameters.  Each firmware implementation uses the subset of parameters
//  needed by that particular version.
//
//  In this example, there are three parameters:  
//  
//     unsigned A, unsigned B, and double C.
//
//  These are used differently by three successive firmware versions:
//
//   - Firmware v1 and v2 use unsigned A and unsigned B.
//   - Firmware v3 uses unsigned A and double C.
//
//  To see by example howto implement a CondFormat class, make sure to see also the files:
//
//  CondFormats/L1TYellow/src/YellowParams.cc               
//  CondFormats/L1TYellow/src/T_EventSetup_YellowParams.cc  
//  CondFormats/L1TYellow/src/classes.h
//  CondFormats/L1TYellow/src/classes_def.xml
//  CondFormats/DataRecord/src/L1TYellowParamsRcd.cc
//  CondFormats/DataRecord/interface/L1TYellowParamsRcd.h
//
//  Note that L1TYellowParamsRcd file includes L1T at beginning of filename, as
//  this is located in the non-L1T specific directory CondFormats/DataRecord.
//

#include <iostream>

namespace l1t {
  
  class YellowParams {
    
  public:
    // Default constructor is required for CondFormats classes:
    YellowParams() {}

    // Const getters for parameters:
    unsigned firmwareVersion() const {return m_fw_version;}
    unsigned paramA() const {return m_a;}
    unsigned paramB() const {return m_b;}
    double   paramC() const {return m_c;}
    
    // Non-const setters:
    void setFirmwareVersion(unsigned v) {m_fw_version=v;};
    void setParamA(unsigned v) {m_a = v;}
    void setParamB(unsigned v) {m_b = v;}
    void setParamC(double   v) {m_c = v;}

    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const YellowParams & y) { y.print(o); return o; } 
    
  private:
    // l1t uses the "m_" convention for marking member data:
    unsigned m_fw_version, m_a, m_b;
    double m_c;
  };

}// namespace
#endif
