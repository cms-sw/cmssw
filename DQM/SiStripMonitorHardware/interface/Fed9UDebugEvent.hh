#ifndef FED9UDEBUGEVENT_H
#define FED9UDEBUGEVENT_H

#include "Fed9UUtils.hh"

namespace Fed9U {
  
  class Fed9UDebugEvent : public Fed9UEvent
  {
    //constructors
  public:
    Fed9UDebugEvent() : Fed9UEvent() {}
    Fed9UDebugEvent(u32 * buffer, const Fed9UDescription * currentDescription,u32 bufsize = 0) 
      : Fed9UEvent(buffer,currentDescription,bufsize) { Init(buffer, currentDescription, bufsize); }
  private:
    //prevent copying as in Fed9UEvent
    Fed9UDebugEvent(const Fed9UEvent &);
    Fed9UDebugEvent(const Fed9UDebugEvent &);
    Fed9UDebugEvent & operator = (const Fed9UEvent &);
    Fed9UDebugEvent & operator = (const Fed9UDebugEvent &);
    //add new Fed9UEventIterator for new functions to use and replace methods which set buffer to also set d_buffer
  public:
    void Init(u32 * buffer, const Fed9UDescription * currentDescription, u32 bufsize) { 
      d_buffer.set(reinterpret_cast<u8*>(buffer), bufsize * 4);
      (static_cast<Fed9UEvent*>(this))->Init(buffer,currentDescription,bufsize);
    }
    void DebugInit(u32 * buffer, const Fed9UDescription * currentDescription, u32 bufsize, std::string& report, u32& readCrc, u32& calcCrc) {
      Init(buffer,currentDescription,bufsize);
      (static_cast<Fed9UEvent*>(this))->DebugInit(buffer,currentDescription,bufsize,report,readCrc,calcCrc);
    }
  private:
    Fed9UEventIterator d_buffer;
    static const u16 d_SPECIAL_OFF = 8;
    bool getBitFSOP(unsigned int bitNumber, unsigned int fpga);
    
    //new methods
  public:
    // Frame Synch Out Packet (0-2, 0-7)
    u32 getFSOP(unsigned int word, unsigned int FENumber);
    // Frame Length (0-7)
    u16 getFLEN(unsigned int FENumber);
    // Back-End Status Register
    u32 getBESR(void) const;
    // Reserved words (0-4)
    u32 getRES(unsigned int WordNumber);
    // DAQ register (0-1)
    u32 getDAQ(unsigned int DaqRegisterNumber);
    
    bool getAPV1Error(unsigned int fpga,unsigned int fiber);
    bool getAPV1WrongHeader(unsigned int fpga,unsigned int fiber);
    bool getAPV2Error(unsigned int fpga,unsigned int fiber);
    bool getAPV2WrongHeader(unsigned int fpga,unsigned int fiber);
    bool getOutOfSync(unsigned int fpga,unsigned int fiber);
    bool getUnlocked(unsigned int fpga,unsigned int fiber);
    
    u16 getFeMajorAddress(unsigned int fpga);
    bool getFeEnabled(unsigned int fpga);
    bool getFeOverflow(unsigned int fpga);
    
    bool getInternalFreeze();
    bool getBXError();

    // TODO: maybe pick this value from the framework
    enum {MinimumBufferSize = 152};
  };

}


#endif
