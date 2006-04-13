#ifndef DTRawToDigi_DTROChainCoding_h
#define DTRawToDigi_DTROChainCoding_h

/** \class DTROChainCoding
 *  A class for handling the DT Read-out chain.
 *
 *  $Date: 2006/03/24 16:14:07 $
 *  $Revision: 1.3 $
 *  \author M. Zanetti - INFN Padova
 */

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>

#include <vector>

#include <boost/cstdint.hpp>

using namespace std;

class DTROChainCoding {

public:
  
  /// Constructors

  DTROChainCoding(): code(0) {}

  DTROChainCoding(const int &  ddu, const int &  ros, 
		  const int &  rob, const int &  tdc, const int &  channel) {    
    code = 
      ddu << DDU_SHIFT | 
      ros << ROS_SHIFT |
      rob << ROB_SHIFT |
      tdc << TDC_SHIFT |
      channel << CHANNEL_SHIFT;
  }

  DTROChainCoding(uint32_t code_): code(code_) {}
  
  /// Destructor
  virtual ~DTROChainCoding() {}

  /// Setters  ///////////////////////
  inline void setCode(const uint32_t & code_) {code = code_;}
  inline void setChain(const int &  ddu, const int &  ros, 
		       const int &  rob, const int &  tdc, const int &  channel) {
    
    code = 
      ddu << DDU_SHIFT | 
      ros << ROS_SHIFT |
      rob << ROB_SHIFT |
      tdc << TDC_SHIFT |
      channel << CHANNEL_SHIFT;
  }
  inline void setDDU(const int & ID) { code = code | (ID << DDU_SHIFT); } 
  inline void setROS(const int & ID) { code = code | (ID << ROS_SHIFT); } 
  inline void setROB(const int & ID) { code = code | (ID << ROB_SHIFT); } 
  inline void setTDC(const int & ID) { code = code | (ID << TDC_SHIFT); } 
  inline void setChannel(const int & ID) { code = code | (ID << CHANNEL_SHIFT); } 
  
  /// Getters ///////////////////////
  inline uint32_t getCode() const { return code; }
  inline int getDDU() const { return (code >> DDU_SHIFT) & DDU_MASK; }
  inline int getDDUID() const { return (code >> DDU_SHIFT) ; }
  inline int getROS() const { return (code >> ROS_SHIFT) & ROS_MASK; }
  inline int getROSID() const { return (code >> ROS_SHIFT) ; }
  inline int getROB() const { return (code >> ROB_SHIFT) & ROB_MASK; }
  inline int getROBID() const { return (code >> ROB_SHIFT) ; }
  inline int getTDC() const { return (code >> TDC_SHIFT) & TDC_MASK; }
  inline int getTDCID() const { return (code >> TDC_SHIFT) ; }
  inline int getChannel() const { return (code >> CHANNEL_SHIFT) & CHANNEL_MASK; }
  inline int getChannelID() const { return (code >> CHANNEL_SHIFT) ; }

private:

  uint32_t code;

  // First shift the bits then apply the mask

  // ddu bit are the last ones. I DONT CARE if the ID is > than 730 (I always get the lsb)
  static const int  DDU_SHIFT = 16;
  static const int  DDU_MASK = 0x8;

  static const int  ROS_SHIFT = 12;
  static const int  ROS_MASK = 0xF;

  static const int  ROB_SHIFT = 7;
  static const int  ROB_MASK = 0x1F;

  static const int  TDC_SHIFT = 5;
  static const int  TDC_MASK = 0x3;

  static const int  CHANNEL_SHIFT = 0;
  static const int  CHANNEL_MASK = 0x1F;


};

#endif
