#ifndef L1Trigger_L1TGlobal_L1TGtObject_h
#define L1Trigger_L1TGlobal_L1TGtObject_h

/**
 * \class GlobalObject
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
enum GlobalObject
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

/// the string to enum and enum to string conversions for GlobalObject

struct L1TGtObjectStringToEnum {
    const char* label;
    GlobalObject value;
};

l1t::GlobalObject l1TGtObjectStringToEnum(const std::string&);
std::string l1TGtObjectEnumToString(const GlobalObject&);

}

#endif /*L1Trigger_L1TGlobal_L1TGtObject_h*/
