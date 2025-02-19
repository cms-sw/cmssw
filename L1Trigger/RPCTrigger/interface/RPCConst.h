#ifndef L1Trigger_RPCConst_h
#define L1Trigger_RPCConst_h

#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif //_STAND_ALONE

#include <string>
#include <map>
#include <vector>
#include <bitset>
/** \class RPCConst
 * 
 * Class contains number of L1RpcTrigger specific
 * constanst, and transforming methods (eg. phi <-> segment number)
 * (Should migrate to DDD?)
 *
 * \author  Marcin Konecki, Warsaw
 *          Artur Kalinowski, Warsaw
 *
 ********************************************************************/

class RPCConst {

public:

  enum {
        ITOW_MIN  = 0,            //!< Minimal number of abs(m_tower_number)
        ITOW_MAX  = 16,           //!< Maximal number of abs(m_tower_number)
        //ITOW_MAX_LOWPT  = 7,      //!< Max m_tower number to which low_pt algorithm is used
        IPT_MAX = 31,             //!< Max pt bin code
        NSTRIPS   = 1152,         //!< m_Number of Rpc strips in phi direction.
        NSEG      = NSTRIPS/8,    //!< m_Number of trigger segments. One segment covers 8 RPC strips 
                                  //!<in referencial plane (hardware 2 or 6(2')
        OFFSET    = 5            //!< Offset of the first trigger phi sector [deg]
   };

  //static const int m_TOWER_COUNT = 16 + 1; //!< Only half of the detector.
  
//-----------------import from L1RpcParameters beg------------------    

    //static const double m_pi = 3.14159265358979;
    static const int m_TOWER_COUNT = 16 + 1; //!< Only half of the detector.
    
    static const int m_PT_CODE_MAX = 31; //!< Pt_code range = 0-m_PT_CODE_MAX
    
    static const int m_LOGPLANES_COUNT = 6; //!< Max Logic Planes Count in trigger towers
    
    static const int m_LOGPLANE1 = 0; //!< The Logic Planes are named starting from '1', but in varoius loop indeks are from '0', that's why always use these consts 
    static const int m_LOGPLANE2 = 1;
    static const int m_LOGPLANE3 = 2;
    static const int m_LOGPLANE4 = 3;
    static const int m_LOGPLANE5 = 4;
    static const int m_LOGPLANE6 = 5;
    
    static const int m_FIRST_PLANE = m_LOGPLANE1; //!< Use ase a first index in loops.
    static const int m_LAST_PLANE  = m_LOGPLANE6; //!< Use ase a last index in loops.
  
  /*
  
    static const int m_TOWER_COUNT = 16 + 1; //!< Only half of the detector.
    
    static const int m_PT_CODE_MAX; //!< Pt_code range = 0-m_PT_CODE_MAX
    
    static const int m_LOGPLANES_COUNT = 6; //!< Max Logic Planes Count in trigger towers
    
    static const int m_LOGPLANE1; //!< The Logic Planes are named starting from '1', but in varoius loop indeks are from '0', that's why always use these consts 
    static const int m_LOGPLANE2;
    static const int m_LOGPLANE3;
    static const int m_LOGPLANE4;
    static const int m_LOGPLANE5;
    static const int m_LOGPLANE6;
    
    static const int m_FIRST_PLANE; //!< Use ase a first index in loops.
    static const int m_LAST_PLANE; //!< Use ase a last index in loops.
    */

    ///Log Planes names.
    static const std::string m_LOGPLANE_STR[];
    
    /// Definition of Logic Cone Sizes - number of Logic m_Strips in each plane
    static const unsigned int m_LOGPLANE_SIZE[m_TOWER_COUNT][m_LOGPLANES_COUNT];
    
    ///Definition of Referenece Plane for each m_Tower.
    static const int m_REF_PLANE[m_TOWER_COUNT];
    
    ///m_Number of Logic Planes existing in each m_Tower.
    static const int m_USED_PLANES_COUNT[m_TOWER_COUNT];
    
    ///m_Number of Logic Planes used for Very Low Pt patterns.
    static const int m_VLPT_PLANES_COUNT[m_TOWER_COUNT];
    
    static const int m_VLPT_CUT = 7; //!< Max Pt code of Very Low Pt patterns.
    
    static const int m_NOT_CONECTED = 99; //!< Denotes Logic m_Strips that is not valid (f.e. in Patterns denotes, that in given plane the pattern is not defined).
    
    
    //-------------------------quallity tab-----------------------------------------
    //should be moved somwhere else
    /*
    typedef std::bitset<m_LOGPLANES_COUNT> TQualityBitset; //for quallity tab
    
    struct bitsetLes : public std::less<TQualityBitset>
    {
      bool operator() (const TQualityBitset& x, const TQualityBitset& y) const
      {
        return(x.count() < y.count());
      }
    };
    
    typedef std::multimap<TQualityBitset, int , bitsetLes> TQualityTab;
    typedef TQualityTab::value_type TQualityTabValueType;
    */
    typedef std::vector<short> TQualityTab;
    typedef std::vector<TQualityTab> TQualityTabsVec;
    //----------------------end quallity tab----------------------------------------
    
    ///The coordinates of Logic Cone: m_Tower, m_LogSector,  m_LogSegment.
    struct l1RpcConeCrdnts {
      int m_Tower;
      int m_LogSector;
      int m_LogSegment;
    
      l1RpcConeCrdnts() {
        m_Tower = 0;
        m_LogSector = 0;
        m_LogSegment = 0;
      }
    
      l1RpcConeCrdnts(int m_tower, int logSector, int logSegment ) {
        m_Tower = m_tower;
        m_LogSector = logSector ;
        m_LogSegment = logSegment;
      }
    
      int getSegmentNum() {
        return m_LogSector * 12 + m_LogSegment;
      }
      
      bool operator < (const l1RpcConeCrdnts& cone) const;
    
      bool operator == (const l1RpcConeCrdnts& cone) const;
    };
    
    
    
    /*
    class RPCMuonGen {
    public:
      int m_RunNum, m_EventNum, m_PtCodeGen;
      double m_EtaGen, m_PhiGen, m_PtGen;
      int m_Sign, m_MaxFiredPlanesCnt;
    
      int possibleTrigger;
};*/
    
    //hardware consts - fixed by board design
    static const unsigned int m_TOWERS_ON_TB_CNT = 4;      //!< Max number of towers covered by one Trugger Board.
    static const unsigned int m_SEGMENTS_IN_SECTOR_CNT = 12;   //!< m_Number of Logic Segments in one Logic Sector, defines also the number of Logic Cones for one Logic Sector of one m_Tower.
    static const unsigned int m_GBPHI_OUT_MUONS_CNT = 4;  //!< m_Number of muon candidates return by Trigger Board's phi Ghost Buster
    static const unsigned int m_GBETA_OUT_MUONS_CNT = 4;  //!< m_Number of muon candidates return by Trigger Board's eta Ghost Buster
    static const unsigned int m_TCGB_OUT_MUONS_CNT = 4;   //!< m_Number of muon candidates return by Trigger Crate's Ghost Buster
    static const unsigned int m_FINAL_OUT_MUONS_CNT = 4;  //!< m_Number of muon candidates return by Final GhostBuster&Sorter
    //const that are dependent on trigger configuration (f.e. TBs cnt in TC)
    //are in RPCTriggerConfiguration
    
    
    ///Converts string to inteager number. If string contains chars, that are not digits, throws RPCException.
    int stringToInt(std::string str);
    
    ///Converts inteager number to string.
    std::string intToString(int number);
  
//-----------------import from L1RpcParameters end------------------  
  ///
  ///Method converts pt [Gev/c] into pt bin number (0, 31).
  ///
  static int iptFromPt(const double pt);

  ///
  ///Method converts pt bin number (0, 31) to pt [GeV/c].
  ///
  static double ptFromIpt(const int ipt);

  ///
  ///Method converts from m_tower number to eta (gives center of m_tower).
  ///
  static double etaFromTowerNum(const int atower);

  ///
  ///Method converts from eta to trigger m_tower number.
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
  ///Method checks if m_tower is in barrel (<ITOW_MAX_LOWPT).
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
  static double vxMuRate(int ptCode);

  static double vxIntegMuRate(int ptCode, double etaFrom, double etaTo);

  static double vxIntegMuRate(int ptCode, int m_tower);

private:
  ///
  ///Matrix with pt bins upper limits.
  ///
  static const double m_pts[RPCConst::IPT_MAX+1];

  ///Matrix with approximate upper towers limits. Only positive ones are reported,
  ///for negative ones mirror symmetry assumed.
  static const double m_etas[RPCConst::ITOW_MAX+2];

 
};
#endif

