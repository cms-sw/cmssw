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
  
//new methods
public:
  u32 getFSOP_8_1(void) const;
  u32 getFSOP_8_2(void) const;
  u16 getFSOP_8_3(void) const;
  u16 getFLEN_8(void) const;
  u32 getBESR(void) const;
  
  u32 getFSOP_7_1(void) const;
  u32 getFSOP_7_2(void) const;
  u16 getFSOP_7_3(void) const;
  u16 getFLEN_7(void) const;
  u32 getRES_5(void) const;
  
  u32 getFSOP_6_1(void) const;
  u32 getFSOP_6_2(void) const;
  u16 getFSOP_6_3(void) const;
  u16 getFLEN_6(void) const;
  u32 getRES_4(void) const;
  
  u32 getFSOP_5_1(void) const;
  u32 getFSOP_5_2(void) const;
  u16 getFSOP_5_3(void) const;
  u16 getFLEN_5(void) const;
  u32 getRES_3(void) const;
  
  u32 getFSOP_4_1(void) const;
  u32 getFSOP_4_2(void) const;
  u16 getFSOP_4_3(void) const;
  u16 getFLEN_4(void) const;
  u32 getRES_2(void) const;
  
  u32 getFSOP_3_1(void) const;
  u32 getFSOP_3_2(void) const;
  u16 getFSOP_3_3(void) const;
  u16 getFLEN_3(void) const;
  u32 getRES_1(void) const;
  
  u32 getFSOP_2_1(void) const;
  u32 getFSOP_2_2(void) const;
  u16 getFSOP_2_3(void) const;
  u16 getFLEN_2(void) const;
  u32 getDAQ_2(void) const;
  
  u32 getFSOP_1_1(void) const;
  u32 getFSOP_1_2(void) const;
  u16 getFSOP_1_3(void) const;
  u16 getFLEN_1(void) const;
  u32 getDAQ_1(void) const;
};

}
