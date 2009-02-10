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

  static std::pair<int,int> getSiPixelFEDIds();
  static std::pair<int,int> getSiStripFEDIds();

  static std::pair<int,int> getPreShowerFEDIds();
  static std::pair<int,int> getEcalFEDIds();
  static std::pair<int,int> getCastorFEDIds();
  static std::pair<int,int> getHcalFEDIds();

  static std::pair<int,int> getLumiScalersFEDIds();

  static std::pair<int,int> getCSCFEDIds();
  static std::pair<int,int> getCSCTFFEDIds();

  static std::pair<int,int> getDTFEDIds();
  static std::pair<int,int> getDTTFFEDIds();

  static std::pair<int,int> getRPCFEDIds();

  static std::pair<int,int> getTriggerGTPFEDIds();
  static std::pair<int,int> getTriggerEGTPFEDIds();

  static std::pair<int,int> getTriggerGCTFEDIds();
  
  static std::pair<int,int> getTriggerLTCmtccFEDIds();
  static std::pair<int,int> getTriggerLTCFEDIds();

  static std::pair<int, int> getTriggerLTCTriggerFEDID();
  static std::pair<int, int> getTriggerLTCHCALFEDID();
  static std::pair<int, int> getTriggerLTCSiStripFEDID();
  static std::pair<int, int> getTriggerLTCECALFEDID();
  static std::pair<int, int> getTriggerLTCTotemCastorFEDID();
  static std::pair<int, int> getTriggerLTCRPCFEDID();
  static std::pair<int, int> getTriggerLTCCSCFEDID();
  static std::pair<int, int> getTriggerLTCDTFEDID();
  static std::pair<int, int> getTriggerLTCSiPixelFEDID();

  static std::pair<int, int> getCSCDDUFEDIds();
  static std::pair<int, int> getCSCContingencyFEDIds();
  static std::pair<int, int> getCSCTFSPFEDIds();

  static std::pair<int, int> getDAQeFEDFEDIds();

  static int lastFEDId();

  static void init();

  static bool inRange(int);
  static bool inRangeNoGT(int);

  static const std::string &fromDet(int);

 private:
  static std::vector<std::string> from_;
  static bool *in_;
  static bool init_;

   enum {
     NOT_A_FEDID = -1,
     MAXFEDID = 1023, // 10 bits
     MINSiPixelFEDID = 0,
     MAXSiPixelFEDID = 39,
     MINSiStripFEDID = 50,
     MAXSiStripFEDID = 489,
     MINPreShowerFEDID = 520,
     MAXPreShowerFEDID = 575,
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
     MAXDTFEDID = 775,
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
     MAXDAQeFEDFEDID = 931
   };
};

#endif // FEDNumbering_H
