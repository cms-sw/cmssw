#ifndef CTPPSDigi_CTPPSDiamondDigi_h
#define CTPPSDigi_CTPPSDiamondDigi_h

/** \class CTPPSDiamondDigi
 *
 * Digi Class for CTPPS Timing Detector
 *  
 *
 * \author Seyed Mohsen Etesami
 * March 2016
 */

#include <cstdint>
#include "DataFormats/CTPPSDigi/interface/HPTDCErrorFlags.h"

class CTPPSDiamondDigi{

  public:
  
    CTPPSDiamondDigi(unsigned int ledgt_, unsigned int tedgt_, unsigned int threvolt, bool mhit_, unsigned short hptdcerror_);
    CTPPSDiamondDigi();
    ~CTPPSDiamondDigi() {};
  
    /// Digis are equal if they are have same  ledt and tedt, threshold voltage, multihit flag, hptdcerror flags
    bool operator==(const CTPPSDiamondDigi& digi) const;

    /// Return digi values number
  
    unsigned int getLeadingEdge() const 
    { 
      return ledgt; 
    }
  
    unsigned int getTrailingEdge() const 
    { 
      return tedgt; 
    }
  
    unsigned int getThresholdVoltage() const 
    { 
      return threvolt; 
    }
  
    bool getMultipleHit() const 
    { 
      return mhit; 
    }
  
    HPTDCErrorFlags getHPTDCErrorFlags() const 
    { 
      return hptdcerror; 
    }

    /// Set digi values
    inline void setLeadingEdge(unsigned int ledgt_) 
    { 
      ledgt = ledgt_; 
    }
    inline void setTrailingEdge(unsigned int tedgt_) 
    { 
      tedgt = tedgt_; 
    }
    inline void setThresholdVoltage(unsigned int threvolt_) 
    { 
      threvolt = threvolt_; 
    }
    inline void setMultipleHit(bool mhit_) 
    { 
      mhit = mhit_; 
    }
    inline void setHPTDCErrorFlags(const HPTDCErrorFlags& hptdcerror_) 
    { 
      hptdcerror = hptdcerror_; 
    }


  private:
    // variable represents leading edge time
    unsigned int ledgt;
    // variable	represents trailing edge time
    unsigned int tedgt;
    // variable represents threshold voltage
    unsigned int threvolt;
    // variable represents multi-hit 
    bool mhit;
    HPTDCErrorFlags hptdcerror;
};

#include <iostream>


inline bool operator< (const CTPPSDiamondDigi& one, const CTPPSDiamondDigi& other)
{
  if( one.getLeadingEdge() < other.getLeadingEdge() )                                     
    return true; 
  if( one.getLeadingEdge() > other.getLeadingEdge() )                                      
    return false;
  if( one.getTrailingEdge() < other.getTrailingEdge() )                                    
    return true;
  if( one.getTrailingEdge() > other.getTrailingEdge() )                                     
    return false;
  if( one.getMultipleHit() < other.getMultipleHit() )                                 
    return true;
  if( one.getMultipleHit() > other.getMultipleHit() )                                    
    return false;
  if( one.getHPTDCErrorFlags().getErrorFlag() < other.getHPTDCErrorFlags().getErrorFlag() ) 
    return true;
  if( one.getHPTDCErrorFlags().getErrorFlag() > other.getHPTDCErrorFlags().getErrorFlag() ) 
    return false;
  if( one.getThresholdVoltage() < other.getThresholdVoltage() )                             
    return true;
  if( one.getThresholdVoltage() > other.getThresholdVoltage() )  
    return false;
  return false;
}  


inline std::ostream & operator<<(std::ostream & o, const CTPPSDiamondDigi& digi)
{
  return o << " " << digi.getLeadingEdge()
	   << " " << digi.getTrailingEdge()
           << " " << digi.getThresholdVoltage()
           << " " << digi.getMultipleHit()
           << " " << digi.getHPTDCErrorFlags().getErrorFlag();
}

#endif

