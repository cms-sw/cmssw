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

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>
#include <string>

// TODO: update the documentation

/**
 *\brief Roman Pot detector ID.
 *
 * There are 3 types of IDs used in CMSSW in the context of RP.
 * \li "class ID" - this class TotemRPDetId, a daughter of DetId
 * \li "raw ID" - uint32_t, the result of rawId() method
 * \li "decimal or symbolic ID" - 4 decimal digit uint32_t, |arm|station|RP|det|
 *
 * The structure of the raw ID is the following (based on the concept of the DetId)
 * Bit 24 = Arm: 1=z>0 0=z<0
 * Bits [22:23] Station
 * Bits [19:21] Roman Pot number
 * Bits [15:18] Si det. number
 *
 * The advantage of the symbolic ID is that it is easily readable and that it can address most of the elements int the RP subdetector system:
 * chip ID = |arm|station|RP|det|VFAT|, ie. 5-digit decimal number (possibly with leading zeros)\n
 * detector ID = |arm|station|RP|det|\n
 * Roman Pot ID =  |arm|station|RP|\n
 * station ID =   |arm|station|\n
 * arm ID =     |arm|\n
 * where
 * \li arm = 0 (left, i.e. z < 0), 1 (right)
 * \li station = 0 (210m), 2 (220m)
 * \li RP = 0 - 5; 0+1 vertical pots (lower |z|), 2+3 horizontal pots, 4+5 vertical pots (higher |z|)
 * \li det = 0 - 9 (0th det has the lowest |z|)
 * \li VFAT = 0 - 4
 *
 * Moreover there is an official naming scheme (EDMS 906715). It is supported by the ...Name() methods.
**/

class TotemRPDetId : public DetId
{  
  public:
    /// Construct from a raw id. It is required that the Detector part of
    /// id is Totem and the SubDet part is RP, otherwise an exception is thrown.
    explicit TotemRPDetId(uint32_t id);
  
    /// Construct from hierarchy indeces.
    TotemRPDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot=0, uint32_t Plane=0, uint32_t Chip=0);

    static const uint32_t totem_rp_subdet_id = 3;
  
    static const uint32_t startArmBit = 24, maskArm = 0x1, maxArm = 1;
    static const uint32_t startStationBit = 22, maskStation = 0x3, maxStation = 2;
    static const uint32_t startRPBit = 19, maskRP = 0x7, maxRP = 5;
    static const uint32_t startPlaneBit = 15, maskPlane = 0xF, maxPlane = 9;
    static const uint32_t startChipBit = 13, maskChip = 0x3, maxChip = 3;
    
    //-------------------- component getters and setters --------------------
     
    uint32_t arm() const
    {
      return ((id_>>startArmBit) & maskArm);
    }

    void setArm(uint32_t arm)
    {
      id_ &= ~(maskArm << startArmBit);
      id_ |= ((arm & maskArm) << startArmBit);
    }

    uint32_t station() const
    {
      return ((id_>>startStationBit) & maskStation);
    }

    void setStation(uint32_t station)
    {
      id_ &= ~(maskStation << startStationBit);
      id_ |= ((station & maskStation) << startStationBit);
    }

    uint32_t rp() const
    {
      return ((id_>>startRPBit) & maskRP);
    }

    void setRP(uint32_t rp)
    {
      id_ &= ~(maskRP << startRPBit);
      id_ |= ((rp & maskRP) << startRPBit);
    }

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

    TotemRPDetId getArmId() const
    {
      TotemRPDetId armId(*this);
      armId.setStation(0);
      armId.setRP(0);
      armId.setPlane(0);
      armId.setChip(0);
      return armId;
    }

    TotemRPDetId getStationId() const
    {
      TotemRPDetId stId(*this);
      stId.setRP(0);
      stId.setPlane(0);
      stId.setChip(0);
      return stId;
    }

    TotemRPDetId getRPId() const
    {
      TotemRPDetId rpId(*this);
      rpId.setPlane(0);
      rpId.setChip(0);
      return rpId;
    }

    TotemRPDetId getPlaneId() const
    {
      TotemRPDetId plId(*this);
      plId.setChip(0);
      return plId;
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
    // NOTE: use in code deprecated!
    
    inline uint32_t getRPDecimalId() const
    {
      return rp() + station()*10 + arm()*100;
    }

    inline uint32_t getPlaneDecimalId() const
    {
      return plane() + getRPDecimalId()*10;
    }

    //-------------------- name methods --------------------

    /// type of name returned by *Name functions
    enum NameFlag { nShort, nFull, nPath };

    std::string subDetectorName(NameFlag flag = nFull) const;
    std::string armName(NameFlag flag = nFull) const;
    std::string stationName(NameFlag flag = nFull) const;
    std::string rpName(NameFlag flag = nFull) const;
    std::string planeName(NameFlag flag = nFull) const;
    std::string chipName(NameFlag flag = nFull) const;
};

std::ostream& operator<<(std::ostream& os, const TotemRPDetId& id);

#endif 
