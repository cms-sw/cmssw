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

#include <boost/cstdint.hpp>
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
    unsigned int ledgt;
    unsigned int tedgt;
    unsigned int threvolt;
    bool mhit;
    HPTDCErrorFlags hptdcerror;
};

inline bool operator< (const CTPPSDiamondDigi& one, const CTPPSDiamondDigi& other)
{
  return one.getLeadingEdge() < other.getLeadingEdge();
}

#include <iostream>

inline std::ostream & operator<<(std::ostream & o, const CTPPSDiamondDigi& digi)
{
  return o << " " << digi.getLeadingEdge()
	   << " " << digi.getTrailingEdge()
           << " " << digi.getThresholdVoltage()
           << " " << digi.getMultipleHit()
           << " " << digi.getHPTDCErrorFlags().error_flags;
}

#endif

