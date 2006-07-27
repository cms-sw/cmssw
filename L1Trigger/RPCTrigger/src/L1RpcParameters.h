/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcParametersH
#define L1RpcParametersH
//------------------------------------------------------------------------------
/**
 * Definitions of constans defining trigger properties for RPC trigger simualtion,
 * don't include directly, but trought L1RpcPacConstsDef.h.
 * A few other constants are defined in L1RpcConst.h.
 * L1RpcTriggerConfiguration.h contains the structure of Trigger Boards, Crates etc.
 * \author Karol Bunkowski (Warsaw)
 */

#include <string>
#include <map>
#include <vector>
#include <bitset>

//----------------------------------------------------------------------------------

namespace rpcparam {

    const int TOWER_COUNT = 16 + 1; //!< Only half of the detector.
    
    const int PT_CODE_MAX = 31; //!< Pt_code range = 0-PT_CODE_MAX
    
    const int LOGPLANES_COUNT = 6; //!< Max Logic Planes Count in trigger towers
    
    const int LOGPLANE1 = 0; //!< The Logic Planes are named starting from '1', but in varoius loop indeks are from '0', that's why always use these consts 
    const int LOGPLANE2 = 1;
    const int LOGPLANE3 = 2;
    const int LOGPLANE4 = 3;
    const int LOGPLANE5 = 4;
    const int LOGPLANE6 = 5;
    
    const int FIRST_PLANE = LOGPLANE1; //!< Use ase a first index in loops.
    const int LAST_PLANE  = LOGPLANE6; //!< Use ase a last index in loops.
    
    
    ///Log Planes names.
    const std::string LOGPLANE_STR[LOGPLANES_COUNT] = {
      "LOGPLANE1", "LOGPLANE2", "LOGPLANE3", "LOGPLANE4", "LOGPLANE5", "LOGPLANE6"
    }; 
    
    /// Definition of Logic Cone Sizes - number of Logic Strips in each plane
    const unsigned int LOGPLANE_SIZE[TOWER_COUNT][LOGPLANES_COUNT] = {
    //LOGPLANE  1,  2,  3   4   5   6
              {72, 56,  8, 40, 40, 24}, //TOWER 0
              {72, 56,  8, 40, 40, 24}, //TOWER 1
              {72, 56,  8, 40, 40, 24}, //TOWER 2
              {72, 56,  8, 40, 40, 24}, //TOWER 3
              {72, 56,  8, 40, 40, 24}, //TOWER 4
              {72, 56, 40,  8, 40, 24}, //TOWER 5
              {56, 72, 40,  8, 24,  0}, //TOWER 6
              {72, 56, 40,  8, 24,  0}, //TOWER 7
              {72, 24, 40,  8,  0,  0}, //TOWER 8
              {72,  8, 40,  0,  0,  0}, //TOWER 9
              {72,  8, 40, 24,  0,  0}, //TOWER 10
              {72,  8, 40, 24,  0,  0}, //TOWER 11
              {72,  8, 40, 24,  0,  0}, //TOWER 12
              {72,  8, 40, 24,  0,  0}, //TOWER 13
              {72,  8, 40, 24,  0,  0}, //TOWER 14
              {72,  8, 40, 24,  0,  0}, //TOWER 15
              {72,  8, 40, 24,  0,  0}  //TOWER 16
    /*
              {48, 32,  8, 20, 20, 24}, //TOWER 0
              {48, 32,  8, 20, 20, 24}, //TOWER 1
              {48, 32,  8, 20, 20, 24}, //TOWER 2
              {56, 32,  8, 20, 20, 24}, //TOWER 3
              {48, 32,  8, 20, 20, 20}, //TOWER 4
              {48, 40, 20,  8, 16, 20}, //TOWER 5
              {48, 40, 16,  8, 16,  0}, //TOWER 6
              {48, 32, 20,  8, 16,  0}, //TOWER 7
              {48, 40, 24,  8,  0,  0}, //TOWER 8
              {32,  8, 40,  0,  0,  0}, //TOWER 9
              {16,  8, 40, 48,  0,  0}, //TOWER 10
              {24,  8, 40, 48,  0,  0}, //TOWER 11
              {24,  8, 40, 48,  0,  0}, //TOWER 12
              {56,  8, 40, 48,  0,  0}, //TOWER 13
              {56,  8, 40, 48,  0,  0}, //TOWER 14
              {56,  8, 40, 48,  0,  0}, //TOWER 15
              {56,  8, 32, 48,  0,  0}  //TOWER 16
    */
    };  ;
    
    ///Definition of Referenece Plane for each Tower.
    const int REF_PLANE[TOWER_COUNT] = {
    //     0,         1,         2,         3,         4,
      LOGPLANE3, LOGPLANE3, LOGPLANE3, LOGPLANE3, LOGPLANE3,
    //     5,         6,         7,         8,
      LOGPLANE4,  LOGPLANE4, LOGPLANE4, LOGPLANE4,
    //     9,         10,       11,        12,        13,        14,         15,        16,
      LOGPLANE2, LOGPLANE2, LOGPLANE2, LOGPLANE2, LOGPLANE2,  LOGPLANE2, LOGPLANE2, LOGPLANE2
    };
    
    ///Number of Logic Planes existing in each Tower.
    const int USED_PLANES_COUNT[TOWER_COUNT] = {
    //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
      6, 6, 6, 6, 6, 6, 5, 5, 4, 3, 4,  4,  4,  4,  4,  4,  4
    };
    
    ///Number of Logic Planes used for Very Low Pt patterns.
    const int VLPT_PLANES_COUNT[TOWER_COUNT] = {
    //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
      4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,  3,  3,  3,  3,  3,  3
    };
    
    const int VLPT_CUT = 7; //!< Max Pt code of Very Low Pt patterns.
    
    const int NOT_CONECTED = 99; //!< Denotes Logic Strips that is not valid (f.e. in Patterns denotes, that in given plane the pattern is not defined).
    
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
    const unsigned int TOWERS_ON_TB_CNT = 4;      //!< Max number of towers covered by one Trugger Board.
    const unsigned int SEGMENTS_IN_SECTOR_CNT = 12;   //!< Number of Logic Segments in one Logic Sector, defines also the number of Logic Cones for one Logic Sector of one Tower.
    const unsigned int GBPHI_OUT_MUONS_CNT = 4;  //!< Number of muon candidates return by Trigger Board's phi Ghost Buster
    const unsigned int GBETA_OUT_MUONS_CNT = 4;  //!< Number of muon candidates return by Trigger Board's eta Ghost Buster
    const unsigned int TCGB_OUT_MUONS_CNT = 4;   //!< Number of muon candidates return by Trigger Crate's Ghost Buster
    const unsigned int FINAL_OUT_MUONS_CNT = 4;  //!< Number of muon candidates return by Final GhostBuster&Sorter
    //const that are dependent on trigger configuration (f.e. TBs cnt in TC)
    //are in L1RpcTriggerConfiguration
    
    
    ///Converts string to inteager number. If string contains chars, that are not digits, throws L1RpcException.
    int StringToInt(std::string str);
    
    ///Converts inteager number to string.
    std::string IntToString(int number);
} // namespace rpcparam 

#endif

