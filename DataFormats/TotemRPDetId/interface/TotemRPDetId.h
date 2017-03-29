/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *	Hubert Niewiadomski
 *	Jan Kašpar (jan.kaspar@gmail.com) 
 *
 ****************************************************************************/

#ifndef DataFormats_TotemRPDetId_TotemRPDetId
#define DataFormats_TotemRPDetId_TotemRPDetId

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
 * \li "raw ID" - unsigned int, the result of rawId() method
 * \li "decimal or symbolic ID" - 4 decimal digit unsigned int, |arm|station|RP|det|
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
    TotemRPDetId();
  
    /// Construct from a raw id. It is required that the Detector part of
    /// id is Totem and the SubDet part is RP, otherwise an exception is thrown.
    explicit TotemRPDetId(uint32_t id);
  
    /// Construct from fully qualified identifier.
    TotemRPDetId(unsigned int Arm, unsigned int Station, unsigned int RomanPot, unsigned int Detector);

    static const unsigned int totem_rp_subdet_id = 3;
  
    static const unsigned int startArmBit = 24, maskArm = 0x1, maxArm = 1;
    static const unsigned int startStationBit = 22, maskStation = 0x3, maxStation = 2;
    static const unsigned int startRPBit = 19, maskRP = 0x7, maxRP = 5;
    static const unsigned int startDetBit = 15, maskDet = 0xF, maxDet = 9;
     
    inline int arm() const
    {
      return int ((id_>>startArmBit) & maskArm);
    }

    inline int station() const
    {
      return int ((id_>>startStationBit) & maskStation);
    }

    inline int romanPot() const
    {
      return int ((id_>>startRPBit) & maskRP);
    }

    inline int detector() const
    {
      return int ((id_>>startDetBit) & maskDet);
    }

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
    
    inline unsigned int detectorDecId() const
    {
      return detector() + romanPot()*10 + station()*100 + arm()*1000;
    }

    //-------------------------------- static members ---------------------------------------
    
    /// returs true it the raw ID is a TOTEM RP one
    static bool check(unsigned int raw)
    {
      return ((raw >> DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
        ((raw >> DetId::kSubdetOffset) & 0x7) == totem_rp_subdet_id;
    }

    /// fast conversion Raw to Decimal ID
    static unsigned int rawToDecId(unsigned int raw)
    {
      return ((raw >> startArmBit) & maskArm) * 1000
        + ((raw >> startStationBit) & maskStation) * 100
        + ((raw >> startRPBit) & maskRP) * 10
        + ((raw >> startDetBit) & maskDet);
    }

    /// fast conversion Decimal to Raw ID
    static unsigned int decToRawId(unsigned int dec)
    {
      unsigned int i = (DetId::VeryForward << DetId::kDetOffset) | (totem_rp_subdet_id << DetId::kSubdetOffset);
      i &= 0xFE000000;
      i |= ((dec % 10) & maskDet) << startDetBit; dec /= 10;
      i |= ((dec % 10) & maskRP) << startRPBit; dec /= 10;
      i |= ((dec % 10) & maskStation) << startStationBit; dec /= 10;
      i |= ((dec % 10) & maskArm) << startArmBit;
      return i;
    }

    /// returns ID of RP for given detector ID ''i''
    static unsigned int rpOfDet(unsigned int i) { return i / 10; }

    /// returns ID of station for given detector ID ''i''
    static unsigned int stOfDet(unsigned int i) { return i / 100; }

    /// returns ID of arm for given detector ID ''i''
    static unsigned int armOfDet(unsigned int i) { return i / 1000; }

    /// returns ID of station for given RP ID ''i''
    static unsigned int stOfRP(unsigned int i) { return i / 10; }

    /// returns ID of arm for given RP ID ''i''
    static unsigned int armOfRP(unsigned int i) { return i / 100; }

    /// returns ID of arm for given station ID ''i''
    static unsigned int armOfSt(unsigned int i) { return i / 10; }
     

    /// is Detector u-detector?
    /// expect symbolic/decimal ID
    static bool isStripsCoordinateUDirection(int Detector)
    {
      return Detector%2;
    }


    /// type of name returned by *Name functions
    enum NameFlag {nShort, nFull, nPath};

    /// level identifier in the RP hierarchy
    enum ElementLevel {lSystem, lArm, lStation, lRP, lPlane, lChip, lStrip};

    /// returns the name of the RP system
    static std::string systemName(NameFlag flag = nFull);

    /// returns official name of an arm characterized by ''id''; if ''full'' is true, prefix rp_ added
    static std::string armName(unsigned int id, NameFlag flag = nFull);

    /// returns official name of a station characterized by ''id''; if ''full'' is true, name of arm is prefixed
    static std::string stationName(unsigned int id, NameFlag flag = nFull);
  
    /// returns official name of a RP characterized by ''id''; if ''full'' is true, name of station is prefixed
    static std::string rpName(unsigned int id, NameFlag flag = nFull);
  
    /// returns official name of a plane characterized by ''id''; if ''full'' is true, name of RP is prefixed
    static std::string planeName(unsigned int id, NameFlag flag = nFull);
  
    /// returns official name of a chip characterized by ''id''; if ''full'' is true, name of plane is prefixed
    static std::string chipName(unsigned int id, NameFlag flag = nFull);
  
    /// returns official name of a strip characterized by ''id'' (of chip) and strip number; if ''full'' is true, name of chip is prefixed
    static std::string stripName(unsigned int id, unsigned char strip, NameFlag flag = nFull);

    /// shortcut to use any of the *Name methods, given the ElementLevel
    static std::string officialName(ElementLevel level, unsigned int id, NameFlag flag = nFull, unsigned char strip = 0);

  private:
    inline void init(unsigned int Arm, unsigned int Station, unsigned int RomanPot, unsigned int Detector);
};

std::ostream& operator<<(std::ostream& os, const TotemRPDetId& id);

#endif 
