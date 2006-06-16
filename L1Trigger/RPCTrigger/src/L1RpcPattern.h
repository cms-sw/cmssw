/******************************************************************************
*                                                                             *
*  Karol Bunkowski                                                            *
*  Warsaw University 2002                                                     *
*                                                                             *
******************************************************************************/
#ifndef L1RpcPatternH
#define L1RpcPatternH
//-----------------------------------------------------------------------------
/** \class L1RpcPattern
 *
 * Definition of single pattern of muon track, i.e. strips range for every plane,
 * muon sign and ptCode, etc.
 * \author Karol Bunkowski (Warsaw)
 */
#include <vector>
//#include "L1Trigger/RPCTrigger/src/L1RpcParametersDef.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
//-----------------------------------------------------------------------------

class L1RpcPattern  {
public:
  //needed types
  /** \class L1RpcLogicalStrip
   * Logical Strip for pattern definition. It may be OR of few Logic Strips of LogCone
   * The strips range is StripFrom - (StripTo-1)
   * If the pattern is not defined for given plane, the valu of StripFrom is NOT_CONECTED.
  */
  class L1RpcLogicalStrip {
  friend class L1RpcPattern;
  private:
    ///First strip in range.
    unsigned char StripFrom;
    ///Next-to-last strip in range.
    unsigned char StripTo;
  };

private:
  ///LogicalStrip for every LogPlane. 
  L1RpcLogicalStrip Strips[RPCParam::LOGPLANES_COUNT];

  ///Muon's sign.
  char Sign;

  ///Muons ptCode.
  char Code;

  ///Number of pattern in PAC's patterns set.
  short Number;

  /** The PAC algorith that should be used for given Pattern.
    * PAT_TYPE_T - Basic (clasic), PAT_TYPE_E - "impoved" (energetic).
    * @see "Pattern Comparator Trigger Algorithm – implementation in FPGA" */
  RPCParam::TPatternType PatternType;

  ///If pattern is of type PAT_TYPE_E, denotes the index of group to which this pattern belongs.
  char RefGroup;

  /** The index of quality table that should be used for given pattern.
   * The quality table is defined at the beginig of each patterns file */
  char QualityTabNumber;

public:
  ///Default Constructor. Empty pattern, no muon, all planes NOT_CONECTED
  L1RpcPattern();

  void SetStripFrom(int logPlane, int stripFrom) {
    Strips[logPlane].StripFrom = stripFrom;
  }
  void SetStripTo(int logPlane, int stripTo) {
    Strips[logPlane].StripTo = stripTo;
  }

  ///First strip in range.
  int GetStripFrom(int logPlane) const { //logic srtip
    return Strips[logPlane].StripFrom;
  }

  ///Next-to-last strip in range.
  int GetStripTo(int logPlane) const {  //logic srtip
    return Strips[logPlane].StripTo;
  }

  ///Returns the stripFrom position w.r.t the first strip in ref plane.
  int GetBendingStripFrom(int logPlane, int tower) {
    if (Strips[logPlane].StripFrom == RPCParam::NOT_CONECTED)
      return  RPCParam::NOT_CONECTED;                                                   //expand
    return Strips[logPlane].StripFrom - Strips[RPCParam::REF_PLANE[tower]].StripFrom - (RPCParam::LOGPLANE_SIZE[tower][logPlane] - RPCParam::LOGPLANE_SIZE[tower][RPCParam::REF_PLANE[tower]])/2;
  }

  ///Returns the stripTo position w.r.t the first strip in ref plane..
  int GetBendingStripTo(int logPlane, int tower) {
    if (Strips[logPlane].StripTo == RPCParam::NOT_CONECTED+1)
      return  RPCParam::NOT_CONECTED;                                                   //expand
    return Strips[logPlane].StripTo - Strips[RPCParam::REF_PLANE[tower]].StripFrom - (RPCParam::LOGPLANE_SIZE[tower][logPlane] - RPCParam::LOGPLANE_SIZE[tower][RPCParam::REF_PLANE[tower]])/2;
  }

  int GetCode() const{
    return Code;
  };

  int GetSign() const{
    return Sign;
  };

  int GetNumber() const{
    return Number;
  };

  RPCParam::TPatternType GetPatternType() const {
    return PatternType;
  };

  int GetRefGroup() const {
    return RefGroup;
  }

  int GetQualityTabNumber() const {
    return QualityTabNumber;
  };

  void SetCode(int a) {
    Code = a;
  };
  void SetSign(int a) {
    Sign = a;
  };
  void SetNumber(int a) {
    Number = a;
  };

  void SetPatternType(RPCParam::TPatternType patternType) {
    PatternType = patternType;
  };

  void SetRefGroup(int refGroup) {
    RefGroup = refGroup;
  }

  void SetQualityTabNumber(int qualityTabNumber ) {
    QualityTabNumber = qualityTabNumber;
  };

};

typedef std::vector<L1RpcPattern> L1RpcPatternsVec;

#endif
