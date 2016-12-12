/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@gmail.com)
*    
****************************************************************************/

#ifndef DataFormats_TotemDigi_TotemVFATStatus
#define DataFormats_TotemDigi_TotemVFATStatus

#include <bitset>
#include <map>

//----------------------------------------------------------------------------------------------------

/**
 * Class which contains information about conversion from RAW to DIGI for a single read-out chip (VFAT).
 */
class TotemVFATStatus
{
  public:
    TotemVFATStatus(uint8_t _cp = 0) : chipPosition(_cp), status(0) {}

    /// VFAT is present in mapping but no data is present int raw event
    inline bool isMissing() const { return status[0]; }

    /// 12-bit hw id from the header of the vfat frame is diffrent from the 16-bit one from hw mapping.
    inline bool isIDMismatch() const { return status[1]; }
    
    /// Footprint error
    inline bool isFootprintError() const { return status[2]; }

    /// CRC error
    inline bool isCRCError() const { return status[3]; }

    /// VFATFrame event number doesn't follow the number derived from DAQ
    inline bool isECProgressError() const { return status[4]; }

    /// BC number is incorrect
    inline bool isBCProgressError() const { return status[5]; }

    /// All channels from that VFAT are not taken into account
    inline bool isFullyMaskedOut() const { return status[6]; }

    /// Some channels from VFAT ale masked out, but not all
    inline bool isPartiallyMaskedOut() const { return status[7]; }

    /// None channels are masked out
    inline bool isNotMasked() const { return !(status[6] || status[7]); }

    inline void setChipPosition(uint8_t _cp) { chipPosition = _cp; }
    
    inline void setMissing() { status[0]=true; }
    inline void setIDMismatch() { status[1]=true; }
    inline void setFootprintError() { status[2]=true; }
    inline void setCRCError() { status[3]=true; }
    inline void setECProgressError() { status[4]=true; }
    inline void setBCProgressError() { status[5] = true; }
    inline void setFullyMaskedOut() { status[6]=true; }
    inline void setPartiallyMaskedOut() { status[7]=true; }
    inline void setNotMasked() { status[6]=status[7]=false; }

    bool isOK() const
    {
	  return !(status[0] || status[1] || status[2] || status[3] || status[4] || status[5]);
	}

    bool operator < (const TotemVFATStatus &cmp) const
    {
      return (status.to_ulong() < cmp.status.to_ulong());
	}
  
    friend std::ostream& operator << (std::ostream& s, const TotemVFATStatus &st);

  private:
    /// describes placement of the VFAT within the detector
    uint8_t chipPosition;

    /// the status bits
    std::bitset<8> status;
};

#endif
