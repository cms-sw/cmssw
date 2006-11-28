/******************************************************************************
*                                                                             *
*  Karol Bunkowski                                                            *
*  Warsaw University 2002                                                     *
*                                                                             *
******************************************************************************/
#ifndef L1RpcPatternH
#define L1RpcPatternH
//-----------------------------------------------------------------------------
/** \class RPCPattern
 *
 * Definition of single pattern of muon track, i.e. strips range for every plane,
 * muon sign and ptCode, etc.
 * \author Karol Bunkowski (Warsaw)
 */
#include <vector>
#include "L1Trigger/RPCTrigger/src/RPCConst.h"
//-----------------------------------------------------------------------------

class RPCPattern  {
public:
  //needed types
  /** \class RPCLogicalStrip
   * Logical Strip for pattern definition. It may be OR of few Logic m_Strips of LogCone
   * The strips range is m_StripFrom - (m_StripTo-1)
   * If the pattern is not defined for given plane, the valu of m_StripFrom is m_NOT_CONECTED.
  */
  class RPCLogicalStrip {
  friend class RPCPattern;
  private:
    ///First strip in range.
    unsigned char m_StripFrom;
    ///Next-to-last strip in range.
    unsigned char m_StripTo;
  };

private:
  ///LogicalStrip for every LogPlane. 
  RPCLogicalStrip m_Strips[RPCConst::m_LOGPLANES_COUNT];

  ///Muon's sign.
  char m_Sign;

  ///Muons ptCode.
  char m_Code;

  ///m_Number of pattern in m_PAC's patterns set.
  short m_Number;

  /** The m_PAC algorith that should be used for given Pattern.
    * PAT_TYPE_T - Basic (clasic), PAT_TYPE_E - "impoved" (energetic).
    * @see "Pattern Comparator Trigger Algorithm – implementation in FPGA" */
  RPCConst::TPatternType m_PatternType;

  ///If pattern is of type PAT_TYPE_E, denotes the index of group to which this pattern belongs.
  char m_RefGroup;

  /** The index of quality table that should be used for given pattern.
   * The quality table is defined at the beginig of each patterns file */
  char m_QualityTabNumber;

public:
  ///Default Constructor. Empty pattern, no muon, all planes m_NOT_CONECTED
  RPCPattern();

  void setStripFrom(int logPlane, int stripFrom);
  
  void setStripTo(int logPlane, int stripTo);

  ///First strip in range.
  int getStripFrom(int logPlane) const;

  ///Next-to-last strip in range.
  int getStripTo(int logPlane) const;

  ///Returns the stripFrom position w.r.t the first strip in ref plane.
  int getBendingStripFrom(int logPlane, int m_tower);

  ///Returns the stripTo position w.r.t the first strip in ref plane..
  int getBendingStripTo(int logPlane, int m_tower);

  int getCode() const;

  int getSign() const;

  int getNumber() const;

  RPCConst::TPatternType getPatternType() const;

  int getRefGroup() const;

  int getQualityTabNumber() const;

  void setCode(int a);
  
  void setSign(int a);
  
  void setNumber(int a);

  void setPatternType(RPCConst::TPatternType patternType);

  void setRefGroup(int refGroup);

  void setQualityTabNumber(int qualityTabNumber );

};

typedef std::vector<RPCPattern> L1RpcPatternsVec;

#endif
