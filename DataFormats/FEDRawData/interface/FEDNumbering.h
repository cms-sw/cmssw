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
  static std::pair<int,int> getHcalFEDIds();

  static std::pair<int,int> getCSCFEDIds();
  static std::pair<int,int> getCSCTFFEDIds();

  static std::pair<int,int> getDTFEDIds();
  static std::pair<int,int> getDTTFFEDIds();

  static std::pair<int,int> getRPCFEDIds();

  static std::pair<int,int> getTriggerGTPFEDIds();
  static std::pair<int,int> getTriggerEGTPFEDIds();
  
  static std::pair<int,int> getTriggerLTCmtccFEDIds();
  static std::pair<int,int> getTriggerLTCFEDIds();

  static int lastFEDId();

  static void init();

  static bool inRange(int);

  static const std::string &fromDet(int);

 private:
  // MAXFEDID intended to be "last valid" id: [0,MAXFEDID] is the valid range
  // all other MAX and MIN are ALSO including the boundaries
  static const int MAXFEDID;

  // e.g. [MINSiPixelFEDID, MAXSiPixelFEDID] are all valid indices
  static const int MINSiPixelFEDID;
  static const int MAXSiPixelFEDID;

  static const int MINSiStripFEDID;
  static const int MAXSiStripFEDID;


  static const int MINCSCFEDID;
  static const int MAXCSCFEDID;
  static const int MINCSCTFFEDID;
  static const int MAXCSCTFFEDID;
  
  static const int MINDTFEDID;
  static const int MAXDTFEDID;
  static const int MINDTTFFEDID;
  static const int MAXDTTFFEDID;
  
  static const int MINRPCFEDID;
  static const int MAXRPCFEDID;


  static const int MINPreShowerFEDID;
  static const int MAXPreShowerFEDID;

  static const int MINECALFEDID;
  static const int MAXECALFEDID;
  
  static const int MINHCALFEDID;
  static const int MAXHCALFEDID;

  
  static const int MINTriggerGTPFEDID;
  static const int MAXTriggerGTPFEDID;
  static const int MINTriggerEGTPFEDID;
  static const int MAXTriggerEGTPFEDID;
  static const int MINTriggerLTCFEDID;
  static const int MAXTriggerLTCFEDID;
  static const int MINTriggerLTCmtccFEDID;
  static const int MAXTriggerLTCmtccFEDID;

  static std::vector<std::string> from_;
  static bool *in_;
  static bool init_;
};

#endif // FEDNumbering_H
