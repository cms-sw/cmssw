#ifndef FEDNumbering_H
#define FEDNumbering_H
/** \class FEDNumbering
 *  
 *  This class holds the fed numbering scheme for the CMS geometry. 
 *  No two feds should have the same id. Each subdetector has a reserved range.
 *  Gaps between range give flexibility to the numbering.
 *
 *  $Date: 2005/09/15 09:00:10 $
 *  $Revision: 1.2 $
 *  \author G. Bruno - CERN, EP Division
 */   


#include <algorithm>

class FEDNumbering {


 public:

  virtual ~FEDNumbering(){};

  static std::pair<int,int> getSiPixelFEDIds();
  static std::pair<int,int> getSiStripFEDIds();

  static std::pair<int,int> getMuBarFEDIds();
  static std::pair<int,int> getMuEndFEDIds();
  static std::pair<int,int> getMRpcFEDIds();

  static std::pair<int,int> getPreShowerFEDIds();

  static std::pair<int,int> getEcalBarrelFEDIds();
  static std::pair<int,int> getEcalEndcapFEDIds();

  static std::pair<int,int> getHcalFEDIds();

  static std::pair<int,int> getTriggerFEDIds();

  static int lastFEDId();

 private:
  static const int MAXFEDID;

  static const int MINSiPixelFEDID;
  static const int MAXSiPixelFEDID;

  static const int MINSiStripFEDID;
  static const int MAXSiStripFEDID;


  static const int MINCSCFEDID;
  static const int MAXCSCFEDID;  
  
  static const int MINDTFEDID;
  static const int MAXDTFEDID;
  
  static const int MINRPCFEDID;
  static const int MAXRPCFEDID;


  static const int MINPreShowerFEDID;
  static const int MAXPreShowerFEDID;

  static const int MINECALBarrelFEDID;
  static const int MAXECALBarrelFEDID;
  
  static const int MINECALEndcapFEDID;
  static const int MAXECALEndcapFEDID;

  
  static const int MINHCALFEDID;
  static const int MAXHCALFEDID;

  
  static const int MINTriggerFEDID;
  static const int MAXTriggerFEDID;


};

#endif // FEDNumbering_H
