#ifndef L1GlobalTrigger_L1GtObject_h
#define L1GlobalTrigger_L1GtObject_h

/**
 * \class L1GtObject
 *
 *
 * Description: define an enumeration of L1 GT objects.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files
#include <string>

// user include files
//   base class

// forward declarations

/// L1 GT objects
///    ObjNull catch all errors
enum L1GtObject : unsigned int {
  Mu,
  NoIsoEG,
  IsoEG,
  CenJet,
  ForJet,
  TauJet,
  ETM,
  ETT,
  HTT,
  HTM,
  JetCounts,
  HfBitCounts,
  HfRingEtSums,
  TechTrig,
  Castor,
  BPTX,
  GtExternal,
  ObjNull
};

/// the string to enum and enum to string conversions for L1GtObject

struct L1GtObjectStringToEnum {
  const char* label;
  L1GtObject value;
};

L1GtObject l1GtObjectStringToEnum(const std::string&);
std::string l1GtObjectEnumToString(const L1GtObject&);

#endif /*L1GlobalTrigger_L1GtObject_h*/
