#ifndef CSCDigi_CSCDigiUtilities_h
#define CSCDigi_CSCDigiUtilities_h

/**\class CSCDigiUtilities
 *
 * A utility class for doing various calculations
 * and conversions with CSC Digi information.
 *
 * $Date: 2006/02/23 23:10:14 $
 * $Revision: 1.1 $
 *
 * \author L. Gray, UF
 */

class CSCCLCTDigi;
class CSCALCTDigi;
class CSCCorrelatedLCTDigi;

class CSCDigiUtilities
{
 public:

  /**
   *\function correlatedQuality()
   * Calculate the quality of a correlated LCT from a CLCT and an ALCT.
   *
   * Description of Qualities:
   * Quality 0    : !CLCT && !ALCT
   * Quality 1    : !CLCT && ALCT && Accelerator
   * Quality 2    :  CLCT && ALCT && Accelerator
   * Quality 3    : !CLCT && ALCT && !Accelerator
   * Quality 4    :  CLCT Distrip && !ALCT
   * Quality 5    :  CLCT Halfstrip && !ALCT
   * Quality 6-10 :  CLCT Distrip patterns, increasing CLCT + ALCT quality
   * Quality 11-15:  CLCT Halfstrip patterns, increasing CLCT + ALCT quality
   *
   * \remark Ported from L1CSCTrigger/L1MuCSCMotherboard.
   */
  static int correlatedQuality(const CSCCLCTDigi&, const CSCALCTDigi&);

  /**
   * \function lctQuality()
   * Calculate or make a good guess at the quality of the ALCT and CLCT from a Correlated LCT
   *
   * \warning This is only exact in a two cases, when Quality == 10 or Quality ==15
   * \warning Other Correlated LCT qualities are degenerate in CLCT and ALCT quality
   */
  static int lctQuality(const CSCCorrelatedLCTDigi&);

  /**
   * \function validALCT()
   * Determine if the ALCT for this Correlated LCT is valid.
   *
   * \remarks Derivable from quality code above.
   */
  static bool validALCT(const CSCCorrelatedLCTDigi&);

  /**
   * \function validCLCT()
   * Determine if the CLCT for this Correlated LCT is valid.
   *
   * \remarks Derivable from quality code above.
   */
  static bool validCLCT(const CSCCorrelatedLCTDigi&);

  /**
   * \function accelerator()
   * Determine if this Correlated LCT has an accelerator flagged ALCT pattern
   *
   * \remarks returns false if no ALCT present.
   */
  static bool accelerator(const CSCCorrelatedLCTDigi&);

   /**
   * \function accelerator()
   * Determine if this ALCT is an accelerator pattern
   *
   */
  static bool accelerator(const CSCALCTDigi&);
};

#endif
