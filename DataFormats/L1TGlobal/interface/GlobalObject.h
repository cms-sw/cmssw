#ifndef DataFormats_L1TGlobal_GlobalObject_h
#define DataFormats_L1TGlobal_GlobalObject_h

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
    gtMinBiasHFP0,
    gtMinBiasHFM0,
    gtMinBiasHFP1,
    gtMinBiasHFM1,
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

#endif
