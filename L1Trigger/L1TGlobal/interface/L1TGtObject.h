#ifndef L1Trigger_L1TGlobal_L1TGtObject_h
#define L1Trigger_L1TGlobal_L1TGtObject_h

/**
 * \class L1TGtObject
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

namespace l1t {

// user include files
//   base class

// forward declarations

/// L1 GT objects
///    ObjNull catch all errors
enum L1TGtObject
{
    gtMu,
    gtEG,
    gtJet,
    gtTau,
    gtETM,
    gtETT,
    gtHTT,
    gtHTM,
    gtETM2,
    gtMinBias,
    gtExternal,
    ObjNull
};

/// the string to enum and enum to string conversions for L1TGtObject

struct L1TGtObjectStringToEnum {
    const char* label;
    L1TGtObject value;
};

l1t::L1TGtObject l1TGtObjectStringToEnum(const std::string&);
std::string l1TGtObjectEnumToString(const L1TGtObject&);

}

#endif /*L1Trigger_L1TGlobal_L1TGtObject_h*/
