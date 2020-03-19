#ifndef L1Trigger_RPCPattern_h
#define L1Trigger_RPCPattern_h
//-----------------------------------------------------------------------------
/** \class RPCPattern
 *
 * Definition of single pattern of muon track, i.e. strips range for every plane,
 * muon sign and ptCode, etc.
 * \author Karol Bunkowski (Warsaw)
 */
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>
//-----------------------------------------------------------------------------

class RPCPattern {
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

    COND_SERIALIZABLE;
  };

  typedef std::vector<RPCPattern> RPCPatVec;

  struct TQuality {
    unsigned char m_FiredPlanes;
    char m_QualityTabNumber;
    char m_QualityValue;
    char m_logsector;
    char m_logsegment;
    signed char m_tower;

    COND_SERIALIZABLE;
  };

  typedef std::vector<TQuality> TQualityVec;

  // use namespace?
  // Moved/duplicated from RPCConst
  //    enum TPatternType {PAT_TYPE_T, PAT_TYPE_E};
  typedef char TPatternType;
  static const TPatternType PAT_TYPE_T = 0;
  static const TPatternType PAT_TYPE_E = 1;
  static const int m_LOGPLANES_COUNT = 6;
  static const int m_FIRST_PLANE = 0;
  static const int m_LAST_PLANE = 5;
  static const int m_NOT_CONECTED = 99;

  ///Default Constructor. Empty pattern, no muon, all planes m_NOT_CONECTED
  RPCPattern();
  RPCPattern(int tower, int sector, int segment);

  void setStripFrom(int logPlane, int stripFrom);

  void setStripTo(int logPlane, int stripTo);

  ///First strip in range.
  int getStripFrom(int logPlane) const;

  ///Next-to-last strip in range.
  int getStripTo(int logPlane) const;

  int getTower() const;
  int getLogSector() const;
  int getLogSegment() const;

  int getCode() const;

  int getSign() const;

  int getNumber() const;

  TPatternType getPatternType() const;

  int getRefGroup() const;

  int getQualityTabNumber() const;

  void setCoords(int tower, int sector, int segment);

  void setCode(int a);

  void setSign(int a);

  void setNumber(int a);

  void setPatternType(TPatternType patternType);

  void setRefGroup(int refGroup);

  void setQualityTabNumber(int qualityTabNumber);

private:
  ///LogicalStrip for every LogPlane.
  RPCLogicalStrip m_Strips[m_LOGPLANES_COUNT];

  // coordinates
  char m_Tower;
  char m_LogSector;
  char m_LogSegment;

  ///Muon's sign.
  char m_Sign;

  ///Muons ptCode.
  char m_Code;

  /** The m_PAC algorith that should be used for given Pattern.
    * PAT_TYPE_T - Basic (clasic), PAT_TYPE_E - "impoved" (economic).
    * @see "Pattern Comparator Trigger Algorithm – implementation in FPGA" */
  TPatternType m_PatternType;

  ///If pattern is of type PAT_TYPE_E, denotes the index of group to which this pattern belongs.
  char m_RefGroup;

  /** The index of quality table that should be used for given pattern.
   * The quality table is defined at the beginig of each patterns file */
  char m_QualityTabNumber;

  ///m_Number of pattern in m_PAC's patterns set.
  short m_Number;

  COND_SERIALIZABLE;
};

#endif
