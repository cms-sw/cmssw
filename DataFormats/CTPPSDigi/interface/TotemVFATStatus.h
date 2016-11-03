/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@gmail.com)
*    
****************************************************************************/

#ifndef DataFormats_CTPPSDigi_TotemVFATStatus
#define DataFormats_CTPPSDigi_TotemVFATStatus

#include <bitset>
#include <map>

//----------------------------------------------------------------------------------------------------

/**
 * Class which contains information about conversion from RAW to DIGI for a single read-out chip (VFAT).
 */
class TotemVFATStatus
{
  public:
    TotemVFATStatus(uint8_t _cp = 0) : chipPosition(_cp), status(0), numberOfClustersSpecified(false), numberOfClusters(0), eventCounter(0)
    {
    }

    /// Chip position
    inline uint8_t getChipPosition() const { return chipPosition; }
    inline void setChipPosition(uint8_t _cp) { chipPosition = _cp; }

    /// VFAT is present in mapping but no data is present int raw event
    inline bool isMissing() const { return status[0]; }
    inline void setMissing(bool val = true) { status[0] = val; }

    /// 12-bit hw id from the header of the vfat frame is diffrent from the 16-bit one from hw mapping
    inline bool isIDMismatch() const { return status[1]; }
    inline void setIDMismatch(bool val = true) { status[1] = val; }
    
    /// Footprint error
    inline bool isFootprintError() const { return status[2]; }
    inline void setFootprintError(bool val = true) { status[2] = val; }

    /// CRC error
    inline bool isCRCError() const { return status[3]; }
    inline void setCRCError(bool val = true) { status[3] = val; }

    /// VFATFrame event number doesn't follow the number derived from DAQ
    inline bool isECProgressError() const { return status[4]; }
    inline void setECProgressError(bool val = true) { status[4] = val; }

    /// BC number is incorrect
    inline bool isBCProgressError() const { return status[5]; }
    inline void setBCProgressError(bool val = true) { status[5] = val; }

    /// All channels from that VFAT are not taken into account
    inline bool isFullyMaskedOut() const { return status[6]; }
    inline void setFullyMaskedOut() { status[6]=true; }

    /// Some channels from VFAT ale masked out, but not all
    inline bool isPartiallyMaskedOut() const { return status[7]; }
    inline void setPartiallyMaskedOut() { status[7]=true; }

    /// No channels are masked out
    inline bool isNotMasked() const { return !(status[6] || status[7]); }
    inline void setNotMasked() { status[6]=status[7]=false; }

    bool isOK() const
    {
	  return !(status[0] || status[1] || status[2] || status[3] || status[4] || status[5]);
	}

    /// number of clusters
    inline bool isNumberOfClustersSpecified() const { return numberOfClustersSpecified; }
    inline void setNumberOfClustersSpecified(bool v) { numberOfClustersSpecified = v; }

    inline uint8_t getNumberOfClusters() const { return numberOfClusters; }
    inline void setNumberOfClusters(uint8_t v) { numberOfClusters = v; }

    bool operator < (const TotemVFATStatus &cmp) const
    {
      return (status.to_ulong() < cmp.status.to_ulong());
	}
  
    friend std::ostream& operator << (std::ostream& s, const TotemVFATStatus &st);
    
    /// event Counter
    inline uint8_t getEC() const {return eventCounter;}
    inline void setEC(const uint8_t ec) { eventCounter = ec; }

  private:
    /// describes placement of the VFAT within the detector
    uint8_t chipPosition;

    /// the status bits
    std::bitset<8> status;

    /// the number of hit clusters before DAQ trimming
    bool numberOfClustersSpecified;
    uint8_t numberOfClusters;
    
    /// event counter in the VFAT frame
    uint8_t eventCounter;
};

#endif
