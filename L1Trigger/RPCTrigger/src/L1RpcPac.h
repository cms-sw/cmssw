/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
/*
todo
sprawdzic konwencje znaku mionu !!!!! (takze w L1RpcMuon0)
*/
#ifndef L1RpcPacH
#define L1RpcPacH

/** \class L1RpcPac
 *
 * Performes Pattern Comparator algorithm for one LogCone. Returns one muon candidate.
 * The algorithm details i.e. patterns list, algorthm type for each pattern,
 * qaulity definition, are set from PAC definition file.
 * \author Karol Bunkowski (Warsaw)
 */

#ifndef _STAND_ALONE
using namespace std;
#endif

#include <vector>
#include <string>
#include <bitset>
#include <map>
#include <list>
#include <iostream>
#include "L1Trigger/RPCTrigger/src/L1RpcLogCone.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPacMuon.h"
//#include "Trigger/L1RpcTrigger/src/L1RpcException.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPattern.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPacBase.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPatternsParser.h"
//------------------------------------------------------------------------------

class L1RpcPac: public L1RpcPacBase {
public:
  /** Constructor required by L1RpcPacManager.
    * @param patFilesDir -  the directory conataing PAC definition file.
    * It should containe file for this PAC, defined by tower, logSector, logSegment,
    * named pacPat_t<tower>sc<logSector>sg<logSegment>.vhd
    * Containers: EnergeticPatternsGroupList and TrackPatternsGroup are
    * filled with patterns from file (the method ParsePatternFile() is called).
    * */
  L1RpcPac(std::string patFilesDir, int tower, int logSector, int logSegment):
    L1RpcPacBase(tower, logSector, logSegment)   { 
    std::string patFileName;

    if(patFilesDir.find("pat") != std::string::npos) { 
      patFileName = patFilesDir + "pacPat_t" + RPCParam::IntToString(ConeCrdnts.Tower) +
          "sc" + RPCParam::IntToString(ConeCrdnts.LogSector) + "sg" + RPCParam::IntToString(ConeCrdnts.LogSegment) + ".xml";

      L1RpcPatternsParser parser;
      parser.Parse(patFileName);
      Init(parser);
    }
    else
      //throw L1RpcException("patFilesDir not contines XML");
      std::cout << "patFilesDir not containes XML" << std::endl;

    TrackPatternsGroup.SetGroupDescription("Track PatternsGroup");
    TrackPatternsGroup.SetGroupDescription("Track PatternsGroup");
  }

  
  ///Destructor.
  ~L1RpcPac() {
  }

  void Init(const L1RpcPatternsParser& parser);

  /** Performs Pattern Comparator algorithm for hits from the cone.
    * Calls the RunTrackPatternsGroup() and RunEnergeticPatternsGroups().
    * @return found track candidate (empty if hits does not fit to eny pattern)*/
  L1RpcPacMuon Run(const L1RpcLogCone& cone) const;

  /** @return true, if logStrip defined by logStripNum and logPlane  belongs to the
    * EPatternsGroup from EnergeticPatternsGroupList defined by groupNum. */
  bool GetEPatternsGroupShape(int groupNum, int logPlane, int logStripNum);

  /** @return true, if logStrip defined by logStripNum and logPlane  belongs to the
    * TrackPatternsGroup. */
  bool GetTPatternsGroupShape(int logPlane, int logStripNum) {
    return TrackPatternsGroup.GroupShape.GetLogStripState(logPlane, logStripNum);
  }

  /** @return the cout of patterns stored in PatternsVec.
  */
  int GetPatternsCount() {
    return PatternsVec.size();
  }

  /** @return pattern stored in PatternsVec.
    * Needed for patterns explorer.*/
  L1RpcPattern GetPattern(int patNum) const {
    if(PatternsVec.size() == 0)
      //throw L1RpcException("GetPattren(): Patterns vec is empty, mayby it was not filled!");
      cout << "GetPattren(): Patterns vec is empty, mayby it was not filled!" << std::endl;
    return PatternsVec[patNum];
  }

  /** @return the count af all patterns gropu, i.e. 1 + EnergeticPatternsGroupList.size(). */
  int GetPatternsGroupCount () {
    return (1 + EnergeticPatternsGroupList.size() ); //1 = track pattrens group
  }

  std::string GetPatternsGroupDescription(int patternGroupNum);

private:
//---------------------needed types------------
  /** \class TPatternsGroup
    * Basic class for storing grouped patterns inside Pac.
    * In group (object of class TPatternsGroup) the patterns belonging to given
    * group are stored in PatternsVec. These patterns are use in trigger algorithm*/
  class TPatternsGroup {
    friend class L1RpcPac;
  protected:
    RPCParam::TPatternType PatternsGroupType;
    //L1RpcPatternsVec PatternsVec; //!< Vector of patterns.
    std::vector<L1RpcPatternsVec::const_iterator> PatternsItVec; //!< Vector of itereator on PatternsVec in Pac.
    L1RpcLogCone GroupShape; //!< Set LogStrips denotes strips beloging to the group.
    std::string GroupDescription;

  public:
    ///The pattern is added to the PatternsVec, the GroupShape is updated (UpdateShape() is called).
    void AddPattern(const L1RpcPatternsVec::const_iterator& pattern) {
      UpdateShape(pattern);
      PatternsItVec.push_back(pattern);
    }

    ///Updates GroupShape, i.e. sets to true strips belonging to the pattern. Coleed in AddPattern()
    void UpdateShape(const L1RpcPatternsVec::const_iterator& pattern); 

    void SetPatternsGroupType(RPCParam::TPatternType patternsGroupType) {
      PatternsGroupType = patternsGroupType;
    }

    RPCParam::TPatternType GetPatternsGroupType() {
      return PatternsGroupType;
    }

    void SetGroupDescription(std::string groupDescription) {
      GroupDescription = groupDescription;
    }

    std::string GetGroupDescription() const {
      return GroupDescription;
    }
  };
  
  /** \class TEPatternsGroup
    * Group of paterns for "improved"("energetic") algorithm.
    * In current implementation all patterns in given group must have the same
    * code and sign. All patterns must have the same QualityTabNumber.
    * Patterns of given code and sign can be devided between a few EPatternsGroups,
    * indexed by RefGroup.
    * The group Code, Sign, RefGroup is definded by pattern index 0 in PatternsVec*/
  class TEPatternsGroup: public TPatternsGroup {
    friend class L1RpcPac;
  public:
    /** Creates new patterns group. The pattern is added to the group and defined
      * its Code, Sign, RefGroup, QualityTabNumber. */
    TEPatternsGroup(const L1RpcPatternsVec::const_iterator& pattern) {
      AddPattern(pattern);
      PatternsGroupType = RPCParam::PAT_TYPE_E;
      QualityTabNumber = pattern->GetQualityTabNumber(); //it is uded in PAC algorithm, so we want to have fast acces.
    };

    ///Checks, if patern can belong to this group, i.e. if has the same Code, Sign, RefGroup and QualityTabNumber.
    bool Check(const L1RpcPatternsVec::const_iterator& pattern);

    ///used for sorting TEPatternsGroups
    bool operator < (const TEPatternsGroup& ePatternsGroup) const;

    /*
    int GetRefGroup() {
      return RefGroup;
    } */
    private:
      short QualityTabNumber;
  };
  //----------------------------------------------------------------------------
  /** \class TTPatternsGroup
    * Group of paterns, for which the "baseline"("track") algorithm is performed. */
  class TTPatternsGroup: public TPatternsGroup {
    friend class L1RpcPac;
  public:
    TTPatternsGroup() {
      PatternsGroupType = RPCParam::PAT_TYPE_T;
    };
  };
  //----------------------------------------------------------------------------
  typedef std::list<TEPatternsGroup> TEPatternsGroupList;
private:
  //Pac parametrs
  short RefGroupCount; //!<From pac file - TT_REF_GROUP_NUMBERS.

  short MaxQuality;

  /** filled only if in constructor L1RpcPac() createPatternsVec == true.
    * Contains all patterns from pac file. Needed for patterns expolorer,
    * does not used in trigger algorithm. */
  L1RpcPatternsVec PatternsVec;
  
  /** The definiton of allowed coincidence of hits in planes and quality values assigned to them.
    * There can be few quality tables in one PAC, to every pattern one of those quality table is asigned.
    * (In barrel usualy 2 quality tables are used: one for high pt (4/6) and one for low pt (3/4).
    * One qaulity table is multimap<biteset, int>, bitset defines allowed coincidance,
    * int defines the quality value. QualityTabsVec is a vector of these maps,
    * the index in vector correspondes to the QualityTabNumber in pattern.
    * @see TQualityBitset, TQualityTab, TQualityTabsVec.
    */
  RPCParam::TQualityTabsVec QualityTabsVec;

  /** Container containig EPatternsGroups. Is filled by InsertPattern() during
    * parsing the pac file ParsePatternFile().
    * (Why list? - Acesing the elements is faster in list than in vector????)*/
  TEPatternsGroupList EnergeticPatternsGroupList;

  ///Here patters used in "baseline" algorith are stored.
  TTPatternsGroup TrackPatternsGroup;

private:
  /** Adds one qaulity record to QualityTabsVec.
    * @param qualityTabNumber - index of QualityTab (index in QualityTabsVec),
    * to which new record should be add.
    * @param qualityBitset - definition of plnaes in coincidance.
    * (qualityBitset[0] == true means, that LogPlane1 should be fired).
    *  @param quality - quality value assigned to given coincidance. */
  void InsertQualityRecord(int qualityTabNumber,
                                unsigned short firedPlanes, short quality);

  /** Adds pattern to TrackPatternsGroup or appropriate group
    * from EnergeticPatternsGroupList. If the appropriate TEPatternsGroup does
    * not exist, it is created.*/
  void InsertPatterns(const L1RpcPatternsVec& pattern);

  /** Runs the "baselie" PAC algorithm. Compares the hits from cone with patterns
   * from TrackPatternsGroup. If many patterns fist to the hits (like usual),
   * the pattern, in which the hits coincidance had highest qauality is chosen
   * Next criteria is higer code.
   * the quality, code, and sign of this pattern are assigned to the returned L1RpcPacMuon. */
  L1RpcPacMuon RunTrackPatternsGroup(const L1RpcLogCone& cone) const;

  /** Runs the "improved" PAC algorithm. Compares the hits from cone with patterns
   * from EnergeticPatternsGroupList. The main diferences from "baselie" PAC algorithm
   * is that here the coincidance (and quality) are searched for LogStrips belonging to the group,
   * and not for every pattern separetly.
   * @see For detailes of algorith see
   * "Pattern Comparator Trigger Algorithm – implementation in FPGA" */
  L1RpcPacMuon RunEnergeticPatternsGroups(const L1RpcLogCone& cone) const;
};
#endif
