// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-

#ifndef ECALSRFLAG
#define ECALSRFLAG

#include "DataFormats/DetId/interface/DetId.h"

/** Base class for Selective Readout Flag (SR flag or SRF).
*/
class EcalSrFlag {
public:
  /** SRP flag for suppression of every channel of the readout
   * unit.
   */
  static const int SRF_SUPPRESS = 0;
  /** SRP flag for zero suppression with level 1.
   */
  static const int SRF_ZS1 = 1;
  /** SRP flag for zero suppression with level 2.
   */
  static const int SRF_ZS2 = 2;
  /** SRP flag for full readout of the readout unit.
   */
  static const int SRF_FULL = 3;
  /** Mask for the 'forced' bit which is set when the decision was
   * forced by an error condition or by configuration.
   * <P>E.g., a force full readout flag has value SRF_FORCED_MASK|SRF_FULL
   */
  static const int SRF_FORCED_MASK = 0x4;

public:
  /** Destructor
   */
  virtual ~EcalSrFlag() {};    

  /** Gets the Det Id the flag is associated to.
   * @return the det id of the readout unit (a barrel TT or a SC).
   */
  virtual const DetId& id() const=0;

  /** SR flag value. See SRF_XXX constants.
   * @return the flag value
   */
  int value() const{ return flag_;}

  /** Set the SR flag value. See SRF_XXX constants.
   * @param flag new flag value. Must be between 0 and 7.
   */
  void setValue(const int& flag) { flag_ = (unsigned char) flag; }

  /** Cast to int: same as value().
   * @return the SR flag value
   */
  operator int() const{
    return flag_;
  }

  /** Return a human readable flag name from its integer value.
   * @param flag the flag value
   * @return the human readable string (which can contain space).
   */
  static std::string flagName(const int& flag){
    return (flag==(flag&0x7))?srfNames[flag]:"Invalid";
  }

  /** Return a human readable flag name from the flag value.
   * @return the human readable string (which can contain space).
   */
  std::string flagName() const{
    return flagName(flag_);
  }
  
protected:
  /** The SRP flag.
   */
  unsigned char flag_;
  
private:
  /** Human readable flag value names
   */
  static const char* const srfNames[];
};
  
#endif //ECALSRFLAG not defined

