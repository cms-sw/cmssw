#ifndef L1RpcConstH
#define L1RpcConstH

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/** \class L1RpcConst
 * 
 * Class contains number of L1RpcTrigger specific
 * constanst, and transforming methods (eg. phi <-> segment number)
 * (Should migrate to DDD?)
 *
 * \author  Marcin Konecki, Warsaw
 *          Artur Kalinowski, Warsaw
 *
 ********************************************************************/

class L1RpcConst {

public:
  enum {
        ITOW_MIN  = 0,            //!< Minimal number of abs(tower_number)
        ITOW_MAX  = 16,           //!< Maximal number of abs(tower_number)
        //ITOW_MAX_LOWPT  = 7,      //!< Max tower number to which low_pt algorithm is used
        IPT_MAX = 31,             //!< Max pt bin code
        NSTRIPS   = 1152,         //!< Number of Rpc strips in phi direction.
        NSEG      = NSTRIPS/8,    //!< Number of trigger segments. One segment covers 8 RPC strips 
                                  //!<in referencial plane (hardware 2 or 6(2')
        OFFSET    = 5            //!< Offset of the first trigger phi sector [deg]
   };

  ///
  ///Method converts pt [Gev/c] into pt bin number (0, 31).
  ///
  static int iptFromPt(const double pt);

  ///
  ///Method converts pt bin number (0, 31) to pt [GeV/c].
  ///
  static double ptFromIpt(const int ipt);

  ///
  ///Method converts from tower number to eta (gives center of tower).
  ///
  static double etaFromTowerNum(const int atower);

  ///
  ///Method converts from eta to trigger tower number.
  ///
  static int   towerNumFromEta(const double eta);

  ///
  ///Method converts from segment number (0, 144).
  ///obsolete
  static double phiFromSegmentNum(const int iseg);

  ///
  ///Method converts from logSegment (0..11) and logSector(0...11) .
  ///
  static double phiFromLogSegSec(const int logSegment, const int logSector);

  ///obsolete
  ///Method converts phi to segment number (0, 144).
  ///
  static int segmentNumFromPhi(const double phi);

  /* obsolete
  ///
  ///Method checks if tower is in barrel (<ITOW_MAX_LOWPT).
  ///
  static int checkBarrel(const int atower);
  */

  /* obsolete
  ///
  ///Matrix with pt thresholds between high, low and very low pt
  ///algorithms.
  static const int IPT_THRESHOLD [2][ITOW_MAX+1];
  */

  ///
  ///Spectrum of muons originating from vertex. See CMS-TN-1995/150
  ///
  static double VxMuRate(int ptCode);

  static double VxIntegMuRate(int ptCode, double etaFrom, double etaTo);

  static double VxIntegMuRate(int ptCode, int tower);

private:
  ///
  ///Matrix with pt bins upper limits.
  ///
  static const double pts[L1RpcConst::IPT_MAX+1];

  ///Matrix with approximate upper towers limits. Only positive ones are reported,
  ///for negative ones mirror symmetry assumed.
  static const double etas[L1RpcConst::ITOW_MAX+2];

 
};
#endif

