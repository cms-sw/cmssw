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

// this class header
#include "DataFormats/L1TGlobal/interface/GlobalObject.h"

// system include files
#include <cstring>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace l1t;


l1t::GlobalObject l1TGtObjectStringToEnum(const std::string& label) {

     
    
    static const l1t::L1TGtObjectStringToEnum l1TGtObjectStringToEnumMap[] = {
            {"Mu",  gtMu},
            {"EG",  gtEG},
	    {"Tau", gtTau},
            {"Jet", gtJet},
            {"ETM", gtETM},
            {"ETT", gtETT},
            {"HTT", gtHTT},
            {"HTM", gtHTM},
	    {"ETM2", gtETM2},
	    {"MinBiasHFP0", gtMinBiasHFP0},
	    {"MinBiasHFM0", gtMinBiasHFM0},
	    {"MinBiasHFP1", gtMinBiasHFP1},
	    {"MinBiasHFM1", gtMinBiasHFM1},
            {"External", gtExternal},
            {"ObjNull", ObjNull},
            {0, (GlobalObject) - 1}
    };

    l1t::GlobalObject value = (GlobalObject) - 1;

    bool found = false;
    for (int i = 0; l1TGtObjectStringToEnumMap[i].label && (!found); ++i)
        if (!std::strcmp(label.c_str(), l1TGtObjectStringToEnumMap[i].label)) {
            found = true;
            value = l1TGtObjectStringToEnumMap[i].value;
        }

    // in case of unrecognized GlobalObject, returns Mu
    // and write a warning (to not throw an exception)
    if (!found) {
        edm::LogInfo("L1TGlobal") << "\n  '" << label
                << "' is not a recognized GlobalObject. \n  Return ObjNull.";

        value = ObjNull;
    }

    if (value == ObjNull) {
        edm::LogInfo("L1TGlobal")
                << "\n  ObjNull means no valid GlobalObject defined!";
    }

    return value;
}

std::string l1t::l1TGtObjectEnumToString(const GlobalObject& gtObject) {

    std::string gtObjectString;

    switch (gtObject) {

        case gtMu: {
            gtObjectString = "Mu";
        }
            break;

        case gtEG: {
            gtObjectString = "EG";
        }
            break;

        case gtTau: {
            gtObjectString = "Tau";
        }
            break;

        case gtJet: {
            gtObjectString = "Jet";
        }
            break;

        case gtETM: {
            gtObjectString = "ETM";
        }
            break;

        case gtETT: {
            gtObjectString = "ETT";
        }
            break;

        case gtHTT: {
            gtObjectString = "HTT";
        }
            break;

        case gtHTM: {
            gtObjectString = "HTM";
        }
            break;

        case gtETM2: {
            gtObjectString = "ETM2";
        }
            break;

        case gtMinBiasHFP0: {
            gtObjectString = "MinBiasHFP0";
        }
            break;

        case gtMinBiasHFM0: {
            gtObjectString = "MinBiasHFM0";
        }
            break;

        case gtMinBiasHFP1: {
            gtObjectString = "MinBiasHFP1";
        }
            break;

        case gtMinBiasHFM1: {
            gtObjectString = "MinBiasHFM1";
        }
            break;

        case gtExternal: {
            gtObjectString = "External";
        }
            break;

        case ObjNull: {
            gtObjectString = "ObjNull";
            edm::LogInfo("L1TGlobal")
                    << "\n  ObjNull means no valid GlobalObject defined!";
        }
            break;

        default: {
            edm::LogInfo("L1TGlobal") << "\n  '" << gtObject
                    << "' is not a recognized GlobalObject. "
                    << "\n  Return ObjNull, which means no valid GlobalObject defined!";

            gtObjectString = "ObjNull";

        }
            break;
    }

    return gtObjectString;

}
