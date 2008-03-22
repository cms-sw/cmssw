#include "DQM/SiStripMonitorHardware/interface/Fed9UDebugEvent.hh"
#include <iomanip>
#define hexpart << setfill('0') << setw(8)


Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP(unsigned int word, unsigned int FENumber) {
  Fed9U::u32 result;
  // TODO: check FENumber to be 0-7
  // Word should be 0-2 
  switch ( word ) {  
  case 0 : 
    result = (d_buffer.getu32(d_SPECIAL_OFF+120-(16*FENumber), true) & 0xFFFFFFFF); 
    break;
  case 1 :
    result = (d_buffer.getu32(d_SPECIAL_OFF+124-(16*FENumber), true) & 0xFFFFFFFF);
    break;
  case 2 :
    result = (d_buffer.getu32(d_SPECIAL_OFF+132-(16*FENumber), true) & 0xFFFF);
    break;
  default : 
    std::cerr << "?!?!" << std::endl;
    result = 0;
  }
  return result;
}

// This retrieves the Front-End buffer length for a given FEUnit
Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN(unsigned int FENumber) {
  // TODO: check FENumber to be 0-7
  return (d_buffer.getu16(d_SPECIAL_OFF+126-(16*FENumber), true) & 0xFFFF);
}

// BackEnd Status Register
Fed9U::u32 Fed9U::Fed9UDebugEvent::getBESR() const {
  return (d_buffer.getu32(d_SPECIAL_OFF+18, true) & 0xFFFFFFFF);
}

// This retreives one of the five reserved words (now unused)
Fed9U::u32 Fed9U::Fed9UDebugEvent::getRES(unsigned int WordNumber) {
  // TODO: check WordNumber to be 0-4
  return (d_buffer.getu32(d_SPECIAL_OFF+98-(16*WordNumber), true) & 0xFFFFFFFF);
}

// This retreives one of the two DAQ registers
Fed9U::u32 Fed9U::Fed9UDebugEvent::getDAQ(unsigned int DaqRegisterNumber) {
  // TODO: check DaqRegisterNumber to be 0-1
  return (d_buffer.getu32(d_SPECIAL_OFF+130-(16*DaqRegisterNumber), true) & 0xFFFFFFFF);
}

bool Fed9U::Fed9UDebugEvent::getAPV1Error(unsigned int fpga,unsigned int fiber) {
  return (!getBitFSOP(fiber*6,fpga));
}

bool Fed9U::Fed9UDebugEvent::getAPV1WrongHeader(unsigned int fpga,unsigned int fiber) {
  return (!getBitFSOP(fiber*6+1,fpga));
}

bool Fed9U::Fed9UDebugEvent::getAPV2Error(unsigned int fpga,unsigned int fiber) {
  return (!getBitFSOP(fiber*6+2,fpga));
}

bool Fed9U::Fed9UDebugEvent::getAPV2WrongHeader(unsigned int fpga,unsigned int fiber) {
  return (!getBitFSOP(fiber*6+3,fpga));
}

bool Fed9U::Fed9UDebugEvent::getOutOfSync(unsigned int fpga,unsigned int fiber) {
  return (!getBitFSOP(fiber*6+4,fpga));
}

bool Fed9U::Fed9UDebugEvent::getUnlocked(unsigned int fpga,unsigned int fiber) {
  return (!getBitFSOP(fiber*6+5,fpga));
}

bool Fed9U::Fed9UDebugEvent::getFeEnabled(unsigned int fpga) {
  return (((getSpecialFeEnableReg()>>(7-fpga))&0x1)==0x1);
}

bool Fed9U::Fed9UDebugEvent::getFeOverflow(unsigned int fpga) {
  return (((getSpecialFeOverflowReg()>>(7-fpga))&0x1)==0x1);
}

Fed9U::u16 Fed9U::Fed9UDebugEvent::getFeMajorAddress(unsigned int fpga) {
  // std::cerr << std::dec << "getFeMajorAddress("<<fpga<<"): " << std::setfill('0') << std::setw(2) << std::hex << ((getFSOP(2,fpga)>>8)&0xFF)<< " "; // debug: TODO: remove this later
  return ((getFSOP(2,fpga)>>8)&0xFF);
}

bool Fed9U::Fed9UDebugEvent::getInternalFreeze() {
  return (((getBESR()>>1)&0x1)==0x1); // TODO: test this
}

bool Fed9U::Fed9UDebugEvent::getBXError() {
  return (((getBESR()>>5)&0x1)==0x1); // TODO: test this
}
 

bool Fed9U::Fed9UDebugEvent::getBitFSOP(unsigned int bitNumber, unsigned int fpga) {
  Fed9U::u32 result = 0;

  if (bitNumber<32)
    // FsopLongLo
    result = (getFSOP(1,fpga) >> bitNumber) & 0x1;
  if ( bitNumber>=32 && bitNumber<64 )
    // FsopLongHi
    result = (getFSOP(0,fpga) >> (bitNumber-32)) & 0x1;
  if ( bitNumber>=64 && bitNumber <80)
    // FsopShort
    result = (getFSOP(2,fpga) >> (bitNumber-64)) & 0x1;

  return (result != 0x0);
}
