#ifndef FEDRawData_FEDNumbering_h
#define FEDRawData_FEDNumbering_h

/** \class FEDNumbering
 *
 *  This class holds the fed numbering scheme for the CMS geometry.
 *  No two feds should have the same id. Each subdetector has a reserved range.
 *  Gaps between ranges give flexibility to the numbering.
 *
 *  $Log
 *
 *  \author G. Bruno - CERN, EP Division
 */

#include <array>

class FEDNumbering {
public:
  static constexpr int lastFEDId() { return MAXFEDID; }

  static bool inRange(int);
  static bool inRangeNoGT(int);

  enum {
    NOT_A_FEDID = -1,
    MAXFEDID = 4096,  // must be larger than largest used FED id
    MINSiPixelFEDID = 0,
    MAXSiPixelFEDID = 40,  // increase from 39 for the pilot blade fed
    MINSiStripFEDID = 50,
    MAXSiStripFEDID = 489,
    MINPreShowerFEDID = 520,
    MAXPreShowerFEDID = 575,
    MINTotemT2FEDID = 577,
    MAXTotemT2FEDID = 577,
    MINTotemRPHorizontalFEDID = 578,
    MAXTotemRPHorizontalFEDID = 581,
    MINCTPPSDiamondFEDID = 582,
    MAXCTPPSDiamondFEDID = 583,
    MINTotemRPVerticalFEDID = 584,
    MAXTotemRPVerticalFEDID = 585,
    MINTotemRPTimingVerticalFEDID = 586,
    MAXTotemRPTimingVerticalFEDID = 587,
    MINECALFEDID = 600,
    MAXECALFEDID = 670,
    MINCASTORFEDID = 690,
    MAXCASTORFEDID = 693,
    MINHCALFEDID = 700,
    MAXHCALFEDID = 731,
    MINLUMISCALERSFEDID = 735,
    MAXLUMISCALERSFEDID = 735,
    MINCSCFEDID = 750,
    MAXCSCFEDID = 757,
    MINCSCTFFEDID = 760,
    MAXCSCTFFEDID = 760,
    MINDTFEDID = 770,
    MAXDTFEDID = 779,
    MINDTTFFEDID = 780,
    MAXDTTFFEDID = 780,
    MINRPCFEDID = 790,
    MAXRPCFEDID = 795,
    MINTriggerGTPFEDID = 812,
    MAXTriggerGTPFEDID = 813,
    MINTriggerEGTPFEDID = 814,
    MAXTriggerEGTPFEDID = 814,
    MINTriggerGCTFEDID = 745,
    MAXTriggerGCTFEDID = 749,
    MINTriggerLTCFEDID = 816,
    MAXTriggerLTCFEDID = 824,
    MINTriggerLTCmtccFEDID = 815,
    MAXTriggerLTCmtccFEDID = 815,
    MINTriggerLTCTriggerFEDID = 816,
    MAXTriggerLTCTriggerFEDID = 816,
    MINTriggerLTCHCALFEDID = 817,
    MAXTriggerLTCHCALFEDID = 817,
    MINTriggerLTCSiStripFEDID = 818,
    MAXTriggerLTCSiStripFEDID = 818,
    MINTriggerLTCECALFEDID = 819,
    MAXTriggerLTCECALFEDID = 819,
    MINTriggerLTCTotemCastorFEDID = 820,
    MAXTriggerLTCTotemCastorFEDID = 820,
    MINTriggerLTCRPCFEDID = 821,
    MAXTriggerLTCRPCFEDID = 821,
    MINTriggerLTCCSCFEDID = 822,
    MAXTriggerLTCCSCFEDID = 822,
    MINTriggerLTCDTFEDID = 823,
    MAXTriggerLTCDTFEDID = 823,
    MINTriggerLTCSiPixelFEDID = 824,
    MAXTriggerLTCSiPixelFEDID = 824,
    MINCSCDDUFEDID = 830,
    MAXCSCDDUFEDID = 869,
    MINCSCContingencyFEDID = 880,
    MAXCSCContingencyFEDID = 887,
    MINCSCTFSPFEDID = 890,
    MAXCSCTFSPFEDID = 901,
    MINDAQeFEDFEDID = 902,
    MAXDAQeFEDFEDID = 931,
    MINMetaDataSoftFEDID = 1022,
    MAXMetaDataSoftFEDID = 1022,
    MINDAQmFEDFEDID = 1023,
    MAXDAQmFEDFEDID = 1023,
    MINTCDSuTCAFEDID = 1024,
    MAXTCDSuTCAFEDID = 1099,
    MINHCALuTCAFEDID = 1100,
    MAXHCALuTCAFEDID = 1199,
    MINSiPixeluTCAFEDID = 1200,
    MAXSiPixeluTCAFEDID = 1349,
    MINRCTFEDID = 1350,
    MAXRCTFEDID = 1359,
    MINCalTrigUp = 1360,
    MAXCalTrigUp = 1367,
    MINDTUROSFEDID = 1369,
    MAXDTUROSFEDID = 1371,
    MINTriggerUpgradeFEDID = 1372,
    MAXTriggerUpgradeFEDID = 1409,
    MINSiPixel2nduTCAFEDID = 1500,
    MAXSiPixel2nduTCAFEDID = 1649,
    MINSiPixelTestFEDID = 1450,
    MAXSiPixelTestFEDID = 1461,
    MINSiPixelAMC13FEDID = 1410,
    MAXSiPixelAMC13FEDID = 1449,
    MINCTPPSPixelsFEDID = 1462,
    MAXCTPPSPixelsFEDID = 1466,
    MINGEMFEDID = 1467,
    MINGE0FEDID = 1473,
    MINGE21FEDID = 1469,
    MAXGEMFEDID = 1478,
    MINDAQvFEDFEDID = 2815,
    MAXDAQvFEDFEDID = 4095
  };
};

#endif  // FEDNumbering_H
