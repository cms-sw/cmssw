#ifndef CTPPSDigi_HPTDCErrorFlags_h
#define CTPPSDigi_HPTDCErrorFlags_h

#include <string>

/** \class HPTDCErrorFlags
 *
 * HPTDC error flags container
 *  
 * \author Seyed Mohsen Etesami
 * \author Laurent Forthomme
 * July 2016
 */

// namespace HPTDCErrors { 
struct HPTDCErrorFlags {
HPTDCErrorFlags(unsigned short flags=0): error_flags(flags) {}

  public:
    bool getErrorId(unsigned short id) 
         {
      switch (id) {
      case 0: return getInternalFatalChipError();
      case 1: return getEventLost();
      case 2: return getHitRejectedByEventSizeLimit();
      case 3: return getHitErrorGroup3();
      case 4: return getHitLostL1OverflowGroup3();
      case 5: return getHitLostROFifoOverflowGroup3();
      case 6: return getHitErrorGroup2();
      case 7: return getHitLostL1OverflowGroup2();
      case 8: return getHitLostROFifoOverflowGroup2();
      case 9: return getHitErrorGroup1();
      case 10: return getHitLostL1OverflowGroup1();
      case 11: return getHitLostROFifoOverflowGroup1();
      case 12: return getHitErrorGroup0();
      case 13: return getHitLostL1OverflowGroup0();
      case 14: return getHitLostROFifoOverflowGroup0();
      default: return true;
      }
    }

    bool getInternalFatalChipError()            { return error_flags&0x1; }
    bool getEventLost()                         { return (error_flags>> 1)&0x1; }
    bool getHitRejectedByEventSizeLimit()       { return (error_flags>> 2)&0x1; }
    bool getHitErrorGroup3()                    { return (error_flags>> 3)&0x1; }
    bool getHitLostL1OverflowGroup3()           { return (error_flags>> 4)&0x1; }
    bool getHitLostROFifoOverflowGroup3()       { return (error_flags>> 5)&0x1; }
    bool getHitErrorGroup2()                    { return (error_flags>> 6)&0x1; }
    bool getHitLostL1OverflowGroup2()           { return (error_flags>> 7)&0x1; }
    bool getHitLostROFifoOverflowGroup2()       { return (error_flags>> 8)&0x1; }
    bool getHitErrorGroup1()                    { return (error_flags>> 9)&0x1; }
    bool getHitLostL1OverflowGroup1()           { return (error_flags>>10)&0x1; }
    bool getHitLostROFifoOverflowGroup1()       { return (error_flags>>11)&0x1; }
    bool getHitErrorGroup0()                    { return (error_flags>>12)&0x1; }
    bool getHitLostL1OverflowGroup0()           { return (error_flags>>13)&0x1; }
    bool getHitLostROFifoOverflowGroup0()       { return (error_flags>>14)&0x1; }
   
    inline unsigned short getErrorFlag() const
    {
      return error_flags;
    }
   
    static std::string getHPTDCErrorName(const unsigned short id)
    {
      switch (id) {
      case 0: return "InternalFatalChipError";
      case 1: return "EventLost";
      case 2: return "HitRejectedByEventSizeLimit";
      case 3: return "HitErrorGroup3";
      case 4: return "HitLostL1OverflowGroup3";
      case 5: return "HitLostROFifoOverflowGroup3";
      case 6: return "HitErrorGroup2";
      case 7: return "HitLostL1OverflowGroup2";
      case 8: return "HitLostROFifoOverflowGroup2";
      case 9: return "HitErrorGroup1";
      case 10: return "HitLostL1OverflowGroup1";
      case 11: return "HitLostROFifoOverflowGroup1";
      case 12: return "HitErrorGroup0";
      case 13: return "HitLostL1OverflowGroup0";
      case 14: return "HitLostROFifoOverflowGroup0";
      default: return "NONE";
      }
    }
  
  private:
    unsigned short error_flags;


};



#endif

