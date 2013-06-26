#ifndef L1Trigger_RPCPacData_h
#define L1Trigger_RPCPacData_h
/*
todo
sprawdzic konwencje znaku mionu !!!!! (takze w L1RpcMuon0)
*/

/** \class RPCPacData
 *
 * Performes Pattern Comparator algorithm for one LogCone. Returns one muon candidate.
 * The algorithm details i.e. patterns list, algorthm type for each pattern,
 * qaulity definition, are set from m_PAC definition file.
 * \author Karol Bunkowski (Warsaw),
 * \author Tomasz Fruboes (Warsaw) - porting to CMSSW
 */

#include <vector>
#include <string>
#include <bitset>
#include <map>
#include <list>
#include <iostream>
#include "CondFormats/L1TObjects/interface/RPCPattern.h"
//#include "L1Trigger/RPCTrigger/interface/RPCPacBase.h"
#include "L1Trigger/RPCTrigger/interface/RPCPatternsParser.h"
#include "L1Trigger/RPCTrigger/interface/TPatternsGroup.h"
#include "L1Trigger/RPCTrigger/interface/TEPatternsGroup.h"
#include "L1Trigger/RPCTrigger/interface/TTPatternsGroup.h"
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"
//------------------------------------------------------------------------------

//class RPCPacData: public RPCPacBase {
class RPCPacData {
   friend class RPCPac;

public:
  
  RPCPacData(std::string patFilesDir, int m_tower, int logSector, int logSegment);
   
  RPCPacData(const RPCPattern::RPCPatVec &patVec, const RPCPattern::TQualityVec &qualVec);

  RPCPacData(const L1RPCConfig * patConf, const int tower, const int sector, const int segment);
  
  void init(const RPCPatternsParser& parser, const RPCConst::l1RpcConeCrdnts& coneCrdnts);

  /*RPCPacMuon run(const RPCLogCone& cone) const;*/

  bool getEPatternsGroupShape(int groupNum, int logPlane, int logStripNum);

  bool getTPatternsGroupShape(int logPlane, int logStripNum);

  int getPatternsCount();

  RPCPattern getPattern(int patNum) const;

  int getPatternsGroupCount();

  std::string getPatternsGroupDescription(int patternGroupNum);

private:
//---------------------needed types------------
  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  typedef std::list<TEPatternsGroup> TEPatternsGroupList;
private:
  //Pac parametrs
  short m_RefGroupCount; //!<From pac file - TT_REF_GROUP_NUMBERS.

  short m_MaxQuality;

  /** filled only if in constructor RPCPacData() createPatternsVec == true.
    * Contains all patterns from pac file. Needed for patterns expolorer,
    * does not used in trigger algorithm. */
  RPCPattern::RPCPatVec m_PatternsVec;
  
  /** The definiton of allowed coincidence of hits in planes and quality values assigned to them.
    * There can be few quality tables in one m_PAC, to every pattern one of those quality table is asigned.
    * (In barrel usualy 2 quality tables are used: one for high pt (4/6) and one for low pt (3/4).
    * One qaulity table is multimap<biteset, int>, bitset defines allowed coincidance,
    * int defines the quality value. m_QualityTabsVec is a vector of these maps,
    * the index in vector correspondes to the m_QualityTabNumber in pattern.
    * @see TQualityBitset, TQualityTab, TQualityTabsVec.
    */
  RPCConst::TQualityTabsVec m_QualityTabsVec;

  /** Container containig EPatternsGroups. Is filled by InsertPattern() during
    * parsing the pac file ParsePatternFile().*/
  TEPatternsGroupList m_EnergeticPatternsGroupList;

  ///Here patters used in "baseline" algorith are stored.
  TTPatternsGroup m_TrackPatternsGroup;

private:
  /** Adds one qaulity record to m_QualityTabsVec.
    * @param qualityTabNumber - index of QualityTab (index in m_QualityTabsVec),
    * to which new record should be add.
    * @param qualityBitset - definition of plnaes in coincidance.
    * (qualityBitset[0] == true means, that LogPlane1 should be fired).
    *  @param quality - quality value assigned to given coincidance. */
  void insertQualityRecord(unsigned int qualityTabNumber,
                                unsigned short firedPlanes, short quality);

  /** Adds pattern to m_TrackPatternsGroup or appropriate group
    * from m_EnergeticPatternsGroupList. If the appropriate TEPatternsGroup does
    * not exist, it is created.*/
  void insertPatterns(const RPCPattern::RPCPatVec &pattern, const int tower = 99, const int sector = 99, const int segment = 99 );

  /** Runs the "baselie" m_PAC algorithm. Compares the hits from cone with patterns
   * from m_TrackPatternsGroup. If many patterns fist to the hits (like usual),
   * the pattern, in which the hits coincidance had highest qauality is chosen
   * Next criteria is higer code.
   * the quality, code, and sign of this pattern are assigned to the returned RPCPacMuon. */
  /*RPCPacMuon runTrackPatternsGroup(const RPCLogCone& cone) const;*/

  /** Runs the "improved" m_PAC algorithm. Compares the hits from cone with patterns
   * from m_EnergeticPatternsGroupList. The main diferences from "baselie" m_PAC algorithm
   * is that here the coincidance (and quality) are searched for LogStrips belonging to the group,
   * and not for every pattern separetly.
   * @see For detailes of algorith see
   * "Pattern Comparator Trigger Algorithm  implementation in FPGA" */
  /*RPCPacMuon runEnergeticPatternsGroups(const RPCLogCone& cone) const;*/
};
#endif
