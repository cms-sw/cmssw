#ifndef L1RpcConstH
#define L1RpcConstH

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <map>
#include <vector>
#include <bitset>
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

  //static const int TOWER_COUNT = 16 + 1; //!< Only half of the detector.
  
//-----------------import from L1RpcParameters beg------------------    
  
    static const int TOWER_COUNT = 16 + 1; //!< Only half of the detector.
    
    static const int PT_CODE_MAX = 31; //!< Pt_code range = 0-PT_CODE_MAX
    
    static const int LOGPLANES_COUNT = 6; //!< Max Logic Planes Count in trigger towers
    
    static const int LOGPLANE1 = 0; //!< The Logic Planes are named starting from '1', but in varoius loop indeks are from '0', that's why always use these consts 
    static const int LOGPLANE2 = 1;
    static const int LOGPLANE3 = 2;
    static const int LOGPLANE4 = 3;
    static const int LOGPLANE5 = 4;
    static const int LOGPLANE6 = 5;
    
    static const int FIRST_PLANE = LOGPLANE1; //!< Use ase a first index in loops.
    static const int LAST_PLANE  = LOGPLANE6; //!< Use ase a last index in loops.
  
  /*
  
    static const int TOWER_COUNT = 16 + 1; //!< Only half of the detector.
    
    static const int PT_CODE_MAX; //!< Pt_code range = 0-PT_CODE_MAX
    
    static const int LOGPLANES_COUNT = 6; //!< Max Logic Planes Count in trigger towers
    
    static const int LOGPLANE1; //!< The Logic Planes are named starting from '1', but in varoius loop indeks are from '0', that's why always use these consts 
    static const int LOGPLANE2;
    static const int LOGPLANE3;
    static const int LOGPLANE4;
    static const int LOGPLANE5;
    static const int LOGPLANE6;
    
    static const int FIRST_PLANE; //!< Use ase a first index in loops.
    static const int LAST_PLANE; //!< Use ase a last index in loops.
    */

    ///Log Planes names.
    static const std::string LOGPLANE_STR[];
    
    /// Definition of Logic Cone Sizes - number of Logic Strips in each plane
    static const unsigned int LOGPLANE_SIZE[TOWER_COUNT][LOGPLANES_COUNT];
    
    ///Definition of Referenece Plane for each Tower.
    static const int REF_PLANE[TOWER_COUNT];
    
    ///Number of Logic Planes existing in each Tower.
    static const int USED_PLANES_COUNT[TOWER_COUNT];
    
    ///Number of Logic Planes used for Very Low Pt patterns.
    static const int VLPT_PLANES_COUNT[TOWER_COUNT];
    
    static const int VLPT_CUT = 7; //!< Max Pt code of Very Low Pt patterns.
    
    static const int NOT_CONECTED = 99; //!< Denotes Logic Strips that is not valid (f.e. in Patterns denotes, that in given plane the pattern is not defined).
    
    /** The PAC algorith that should be used for given Pattern.
      * PAT_TYPE_T - Basic (clasic), PAT_TYPE_E - "impoved" (energetic),
      * @see "Pattern Comparator Trigger Algorithm – implementation in FPGA"
      */
    enum TPatternType {PAT_TYPE_T, PAT_TYPE_E};
    
    
    //-------------------------quallity tab-----------------------------------------
    //should be moved somwhere else
    /*
    typedef std::bitset<LOGPLANES_COUNT> TQualityBitset; //for quallity tab
    
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
    
    ///The coordinates of Logic Cone: Tower, LogSector,  LogSegment.
    struct L1RpcConeCrdnts {
      int Tower;
      int LogSector;
      int LogSegment;
    
      L1RpcConeCrdnts() {
        Tower = 0;
        LogSector = 0;
        LogSegment = 0;
      }
    
      L1RpcConeCrdnts(int tower, int logSector, int logSegment ) {
        Tower = tower;
        LogSector = logSector ;
        LogSegment = logSegment;
      }
    
      int GetSegmentNum() {
        return LogSector * 12 + LogSegment;
      }
      
      bool operator < (const L1RpcConeCrdnts& cone) const;
    
      bool operator == (const L1RpcConeCrdnts& cone) const;
    };
    
    
    
    
    class L1RpcMuonGen {
    public:
      int RunNum, EventNum, PtCodeGen;
      double EtaGen, PhiGen, PtGen;
      int Sign, MaxFiredPlanesCnt;
    
      int PossibleTrigger;
    };
    
    //hardware consts - fixed by board design
    static const unsigned int TOWERS_ON_TB_CNT = 4;      //!< Max number of towers covered by one Trugger Board.
    static const unsigned int SEGMENTS_IN_SECTOR_CNT = 12;   //!< Number of Logic Segments in one Logic Sector, defines also the number of Logic Cones for one Logic Sector of one Tower.
    static const unsigned int GBPHI_OUT_MUONS_CNT = 4;  //!< Number of muon candidates return by Trigger Board's phi Ghost Buster
    static const unsigned int GBETA_OUT_MUONS_CNT = 4;  //!< Number of muon candidates return by Trigger Board's eta Ghost Buster
    static const unsigned int TCGB_OUT_MUONS_CNT = 4;   //!< Number of muon candidates return by Trigger Crate's Ghost Buster
    static const unsigned int FINAL_OUT_MUONS_CNT = 4;  //!< Number of muon candidates return by Final GhostBuster&Sorter
    //const that are dependent on trigger configuration (f.e. TBs cnt in TC)
    //are in L1RpcTriggerConfiguration
    
    
    ///Converts string to inteager number. If string contains chars, that are not digits, throws L1RpcException.
    int StringToInt(std::string str);
    
    ///Converts inteager number to string.
    std::string IntToString(int number);
  
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

