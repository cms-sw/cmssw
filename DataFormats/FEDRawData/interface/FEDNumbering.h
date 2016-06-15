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

#include <vector>
#include <string>

class FEDNumbering {


 public:

  virtual ~FEDNumbering(){};

  static int lastFEDId();

  static void init();

  static bool inRange(int);
  static bool inRangeNoGT(int);

  static const std::string &fromDet(int);

   enum {
     NOT_A_FEDID = -1,
     MAXFEDID = 4096, // must be larger than largest used FED id
     MINSiPixelFEDID = 0,
     MAXSiPixelFEDID = 40,  // increase from 39 for the pilot blade fed
     MINSiStripFEDID = 50,
     MAXSiStripFEDID = 489,
     MINPreShowerFEDID = 520,
     MAXPreShowerFEDID = 575,
     MINTotemTriggerFEDID = 577,
     MAXTotemTriggerFEDID = 577,
     MINTotemRPFEDID = 578,
     MAXTotemRPFEDID = 581,
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
     MAXTriggerEGTPFEDID = 815,
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
     MINDAQmFEDFEDID = 1023,
     MAXDAQmFEDFEDID = 1023,
     MINTCDSuTCAFEDID = 1024,
     MAXTCDSuTCAFEDID = 1099,
     MINHCALuTCAFEDID = 1100,
     MAXHCALuTCAFEDID = 1199,
     MINSiPixeluTCAFEDID = 1200,
     MAXSiPixeluTCAFEDID = 1349,
     MINTriggerUpgradeFEDID = 1350,
     MAXTriggerUpgradeFEDID = 1409,
     MINDAQvFEDFEDID = 2815,
     MAXDAQvFEDFEDID = 4095
   };
 private:
  static std::vector<std::string> from_;
  static bool *in_;
  static bool init_;

};

#endif // FEDNumbering_H
