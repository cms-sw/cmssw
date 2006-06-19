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

  void SetStripFrom(int logPlane, int stripFrom);
  
  void SetStripTo(int logPlane, int stripTo);

  ///First strip in range.
  int GetStripFrom(int logPlane) const;

  ///Next-to-last strip in range.
  int GetStripTo(int logPlane) const;

  ///Returns the stripFrom position w.r.t the first strip in ref plane.
  int GetBendingStripFrom(int logPlane, int tower);

  ///Returns the stripTo position w.r.t the first strip in ref plane..
  int GetBendingStripTo(int logPlane, int tower);

  int GetCode() const;

  int GetSign() const;

  int GetNumber() const;

  RPCParam::TPatternType GetPatternType() const;

  int GetRefGroup() const;

  int GetQualityTabNumber() const;

  void SetCode(int a);
  
  void SetSign(int a);
  
  void SetNumber(int a);

  void SetPatternType(RPCParam::TPatternType patternType);

  void SetRefGroup(int refGroup);

  void SetQualityTabNumber(int qualityTabNumber );

};

typedef std::vector<L1RpcPattern> L1RpcPatternsVec;

#endif
