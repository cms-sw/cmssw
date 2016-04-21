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

// this class header
#include "L1Trigger/L1TGlobal/interface/L1TGtObject.h"

// system include files
#include <cstring>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace l1t;


l1t::L1TGtObject l1TGtObjectStringToEnum(const std::string& label) {

     
    
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
	    {"MinBias", gtMinBias},
            {"External", gtExternal},
            {"ObjNull", ObjNull},
            {0, (L1TGtObject) - 1}
    };

    l1t::L1TGtObject value = (L1TGtObject) - 1;

    bool found = false;
    for (int i = 0; l1TGtObjectStringToEnumMap[i].label && (!found); ++i)
        if (!std::strcmp(label.c_str(), l1TGtObjectStringToEnumMap[i].label)) {
            found = true;
            value = l1TGtObjectStringToEnumMap[i].value;
        }

    // in case of unrecognized L1TGtObject, returns Mu
    // and write a warning (to not throw an exception)
    if (!found) {
        edm::LogInfo("L1TGlobal") << "\n  '" << label
                << "' is not a recognized L1TGtObject. \n  Return ObjNull.";

        value = ObjNull;
    }

    if (value == ObjNull) {
        edm::LogInfo("L1TGlobal")
                << "\n  ObjNull means no valid L1TGtObject defined!";
    }

    return value;
}

std::string l1t::l1TGtObjectEnumToString(const L1TGtObject& gtObject) {

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

        case gtMinBias: {
            gtObjectString = "MinBias";
        }
            break;

        case gtExternal: {
            gtObjectString = "External";
        }
            break;

        case ObjNull: {
            gtObjectString = "ObjNull";
            edm::LogInfo("L1TGlobal")
                    << "\n  ObjNull means no valid L1TGtObject defined!";
        }
            break;

        default: {
            edm::LogInfo("L1TGlobal") << "\n  '" << gtObject
                    << "' is not a recognized L1TGtObject. "
                    << "\n  Return ObjNull, which means no valid L1TGtObject defined!";

            gtObjectString = "ObjNull";

        }
            break;
    }

    return gtObjectString;

}
