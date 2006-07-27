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
 * \author Karol Bunkowski (Warsaw),
 * \author Tomasz Fruboes (Warsaw) - porting to CMSSW
 */

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
#include "L1Trigger/RPCTrigger/src/TPatternsGroup.h"
#include "L1Trigger/RPCTrigger/src/TEPatternsGroup.h"
#include "L1Trigger/RPCTrigger/src/TTPatternsGroup.h"

//------------------------------------------------------------------------------

class L1RpcPac: public L1RpcPacBase {
public:
  
  L1RpcPac(std::string patFilesDir, int tower, int logSector, int logSegment);

   

  void Init(const L1RpcPatternsParser& parser);

  L1RpcPacMuon Run(const L1RpcLogCone& cone) const;

  bool GetEPatternsGroupShape(int groupNum, int logPlane, int logStripNum);

  bool GetTPatternsGroupShape(int logPlane, int logStripNum);

  int GetPatternsCount();

  L1RpcPattern GetPattern(int patNum) const;

  int GetPatternsGroupCount ();

  std::string GetPatternsGroupDescription(int patternGroupNum);

private:
//---------------------needed types------------
  //----------------------------------------------------------------------------
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
  L1RpcConst::TQualityTabsVec QualityTabsVec;

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
  void InsertQualityRecord(unsigned int qualityTabNumber,
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
