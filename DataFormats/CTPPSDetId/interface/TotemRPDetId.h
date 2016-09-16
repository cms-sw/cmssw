/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *	Hubert Niewiadomski
 *	Jan Kašpar (jan.kaspar@gmail.com) 
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSDetId_CTPPSDetId
#define DataFormats_CTPPSDetId_CTPPSDetId

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>
#include <string>

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
    TotemRPDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot=0, uint32_t Detector=0, uint32_t Chip=0);

    static const uint32_t totem_rp_subdet_id = 3;
  
    static const uint32_t startArmBit = 24, maskArm = 0x1, maxArm = 1;
    static const uint32_t startStationBit = 22, maskStation = 0x3, maxStation = 2;
    static const uint32_t startRPBit = 19, maskRP = 0x7, maxRP = 5;
    static const uint32_t startDetBit = 15, maskDet = 0xF, maxDet = 9;
    static const uint32_t startChipBit = 13, maskChip = 0x3, maxChip = 3;
     
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

    uint32_t romanPot() const
    {
      return ((id_>>startRPBit) & maskRP);
    }

    void setRomanPot(uint32_t rp)
    {
      id_ &= ~(maskRP << startRPBit);
      id_ |= ((rp & maskRP) << startRPBit);
    }

    uint32_t detector() const
    {
      return ((id_>>startDetBit) & maskDet);
    }

    void setDetector(uint32_t det)
    {
      id_ &= ~(maskDet << startDetBit);
      id_ |= ((det & maskDet) << startDetBit);
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

    // TODO: needed, remove ??
    int rpCopyNumber() const
    {
      return romanPot() + 10*station() + 100*arm();
    }

    bool isStripsCoordinateUDirection() const
    {
      return detector()%2;
    }

    bool isStripsCoordinateVDirection() const
    {
      return !isStripsCoordinateUDirection();
    }
    
    inline uint32_t detectorDecId() const
    {
      return detector() + romanPot()*10 + station()*100 + arm()*1000;
    }

    //-------------------------------- static members ---------------------------------------
    
    // TODO: needed, remove ??
    /// returs true it the raw ID is a TOTEM RP one
    static bool check(uint32_t raw)
    {
      return ((raw >> DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
        ((raw >> DetId::kSubdetOffset) & 0x7) == totem_rp_subdet_id;
    }

    /// TODO: remove
    /// fast conversion Raw to Decimal ID
    static uint32_t rawToDecId(uint32_t raw)
    {
      return ((raw >> startArmBit) & maskArm) * 1000
        + ((raw >> startStationBit) & maskStation) * 100
        + ((raw >> startRPBit) & maskRP) * 10
        + ((raw >> startDetBit) & maskDet);
    }

    /// TODO: remove
    /// fast conversion Decimal to Raw ID
    static uint32_t decToRawId(uint32_t dec)
    {
      uint32_t i = (DetId::VeryForward << DetId::kDetOffset) | (totem_rp_subdet_id << DetId::kSubdetOffset);
      i &= 0xFE000000;
      i |= ((dec % 10) & maskDet) << startDetBit; dec /= 10;
      i |= ((dec % 10) & maskRP) << startRPBit; dec /= 10;
      i |= ((dec % 10) & maskStation) << startStationBit; dec /= 10;
      i |= ((dec % 10) & maskArm) << startArmBit;
      return i;
    }

    /// TODO: remove all below
    /// returns ID of RP for given detector ID ''i''
    static uint32_t rpOfDet(uint32_t i) { return i / 10; }

    /// returns ID of station for given detector ID ''i''
    static uint32_t stOfDet(uint32_t i) { return i / 100; }

    /// returns ID of arm for given detector ID ''i''
    static uint32_t armOfDet(uint32_t i) { return i / 1000; }

    /// returns ID of station for given RP ID ''i''
    static uint32_t stOfRP(uint32_t i) { return i / 10; }

    /// returns ID of arm for given RP ID ''i''
    static uint32_t armOfRP(uint32_t i) { return i / 100; }

    /// returns ID of arm for given station ID ''i''
    static uint32_t armOfSt(uint32_t i) { return i / 10; }
     

    /// TODO: make non-static
    /// is Detector u-detector?
    /// expect symbolic/decimal ID
    static bool isStripsCoordinateUDirection(int Detector)
    {
      return Detector%2;
    }

    /// TODO: make non-static all below

    /// type of name returned by *Name functions
    enum NameFlag {nShort, nFull, nPath};

    /// level identifier in the RP hierarchy
    enum ElementLevel {lSystem, lArm, lStation, lRP, lPlane, lChip, lStrip};

    /// returns the name of the RP system
    static std::string systemName(NameFlag flag = nFull);

    /// returns official name of an arm characterized by ''id''; if ''full'' is true, prefix rp_ added
    static std::string armName(uint32_t id, NameFlag flag = nFull);

    /// returns official name of a station characterized by ''id''; if ''full'' is true, name of arm is prefixed
    static std::string stationName(uint32_t id, NameFlag flag = nFull);
  
    /// returns official name of a RP characterized by ''id''; if ''full'' is true, name of station is prefixed
    static std::string rpName(uint32_t id, NameFlag flag = nFull);
  
    /// returns official name of a plane characterized by ''id''; if ''full'' is true, name of RP is prefixed
    static std::string planeName(uint32_t id, NameFlag flag = nFull);
  
    /// returns official name of a chip characterized by ''id''; if ''full'' is true, name of plane is prefixed
    static std::string chipName(uint32_t id, NameFlag flag = nFull);
  
    /// returns official name of a strip characterized by ''id'' (of chip) and strip number; if ''full'' is true, name of chip is prefixed
    static std::string stripName(uint32_t id, unsigned char strip, NameFlag flag = nFull);

    /// shortcut to use any of the *Name methods, given the ElementLevel
    static std::string officialName(ElementLevel level, uint32_t id, NameFlag flag = nFull, unsigned char strip = 0);
};

std::ostream& operator<<(std::ostream& os, const TotemRPDetId& id);

#endif 
