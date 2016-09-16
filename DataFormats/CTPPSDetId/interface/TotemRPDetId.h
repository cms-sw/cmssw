/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *	Hubert Niewiadomski
 *	Jan Kašpar (jan.kaspar@gmail.com) 
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSDetId_TotemRPDetId
#define DataFormats_CTPPSDetId_TotemRPDetId

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>
#include <string>

/**
 *\brief Detector ID class for TOTEM Si strip detectors.
 *
 * Beyond the bit assignment in CTPPSDetId, this is the additional structure:
 * Bits [15:18] => plane number from 0 (most near) to 9 (most far)
 * Bits [13:14] => chip (VFAT) number
 * Bits [0:12] => not assigned
**/

class TotemRPDetId : public CTPPSDetId
{  
  public:
    /// Construct from a raw id. It is required that the Detector part of
    /// id is Totem and the SubDet part is RP, otherwise an exception is thrown.
    explicit TotemRPDetId(uint32_t id);

    TotemRPDetId(const CTPPSDetId &id) : CTPPSDetId(id)
    {
    }
  
    /// Construct from hierarchy indeces.
    TotemRPDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot=0, uint32_t Plane=0, uint32_t Chip=0);

    static const uint32_t startPlaneBit = 15, maskPlane = 0xF, maxPlane = 9, lowMaskPlane = 0x7FFF;
    static const uint32_t startChipBit = 13, maskChip = 0x3, maxChip = 3, lowMaskChip = 0x1FFF;
    
    //-------------------- component getters and setters --------------------
     
    uint32_t plane() const
    {
      return ((id_>>startPlaneBit) & maskPlane);
    }

    void setPlane(uint32_t det)
    {
      id_ &= ~(maskPlane << startPlaneBit);
      id_ |= ((det & maskPlane) << startPlaneBit);
    }

    uint32_t chip() const
    {
      return ((id_>>startChipBit) & maskChip);
    }

    void setChip(uint32_t chip)
    {
      id_ &= ~(maskChip << startChipBit);
      id_ |= ((chip & maskChip) << startChipBit);
    }

    //-------------------- id getters for higher-level objects --------------------

    TotemRPDetId getPlaneId() const
    {
      return TotemRPDetId( rawId() & (~lowMaskPlane) );
    }

    //-------------------- strip orientation methods --------------------

    bool isStripsCoordinateUDirection() const
    {
      return plane() % 2;
    }

    bool isStripsCoordinateVDirection() const
    {
      return !isStripsCoordinateUDirection();
    }

    //-------------------- conversions to the obsolete decimal representation --------------------
    // NOTE: only for backward compatibility, do not use otherwise!
    
    inline uint32_t getRPDecimalId() const
    {
      return rp() + station()*10 + arm()*100;
    }

    inline uint32_t getPlaneDecimalId() const
    {
      return plane() + getRPDecimalId()*10;
    }

    //-------------------- name methods --------------------

    std::string planeName(NameFlag flag = nFull) const;
    std::string chipName(NameFlag flag = nFull) const;
};

std::ostream& operator<<(std::ostream& os, const TotemRPDetId& id);

#endif 
