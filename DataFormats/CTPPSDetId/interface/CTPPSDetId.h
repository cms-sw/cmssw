/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
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
 *\brief Base class for CTPPS detector IDs.
 *
 * The bit structure is as follows:
 *   bits [24:24] => arm: 0 (sector 45), 1 (sector 56)
 *   bits [22:23] => station: 0 (210m), 1 (cylyndrical pots), 2 (220m)
 *   bits [19:21] => Roman Pot: 0 (near top), 1 (near bottom), 2 (near horizontal), 3 (far horizontal), 4 (far top), 5 (far bottom)
 *   bits [0:18] => available for derived classes 
 *
 * The ...Name() methods implement the official naming scheme based on EDMS 906715.
**/

class CTPPSDetId : public DetId
{  
  public:
    /// CTPPS sub-detectors
    enum SubDetector { sdTrackingStrip = 3, sdTrackingPixel = 4, sdTimingDiamond = 5, sdTimingFastSilicon = 6 };

    /// Construct from a raw id.
    explicit CTPPSDetId(uint32_t id);
  
    /// Construct from hierarchy indeces.
    CTPPSDetId(uint32_t SubDet, uint32_t Arm, uint32_t Station, uint32_t RomanPot=0);

    //-------------------- bit assignment --------------------
  
    static const uint32_t startArmBit = 24, maskArm = 0x1, maxArm = 1, lowMaskArm = 0xFFFFFF;
    static const uint32_t startStationBit = 22, maskStation = 0x3, maxStation = 2, lowMaskStation = 0x3FFFFF;
    static const uint32_t startRPBit = 19, maskRP = 0x7, maxRP = 5, lowMaskRP = 0x7FFFF;
    
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

    //-------------------- id getters for higher-level objects --------------------

    CTPPSDetId getArmId() const
    {
      return CTPPSDetId( rawId() & (~lowMaskArm) );
    }

    CTPPSDetId getStationId() const
    {
      return CTPPSDetId( rawId() & (~lowMaskStation) );
    }

    CTPPSDetId getRPId() const
    {
      return CTPPSDetId( rawId() & (~lowMaskRP) );
    }

    //-------------------- name methods --------------------

    /// type of name returned by *Name functions
    enum NameFlag { nShort, nFull, nPath };

    std::string subDetectorName(NameFlag flag = nFull) const;
    std::string armName(NameFlag flag = nFull) const;
    std::string stationName(NameFlag flag = nFull) const;
    std::string rpName(NameFlag flag = nFull) const;
};

std::ostream& operator<<(std::ostream& os, const CTPPSDetId& id);

#endif 
