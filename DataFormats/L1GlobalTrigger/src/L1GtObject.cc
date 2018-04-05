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

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtObject.h"

// system include files
#include <cstring>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1GtObject l1GtObjectStringToEnum(const std::string& label) {

    static const L1GtObjectStringToEnum l1GtObjectStringToEnumMap[] = {
            {"Mu", Mu},
            {"NoIsoEG", NoIsoEG},
            {"IsoEG", IsoEG},
            {"CenJet", CenJet},
            {"ForJet", ForJet},
            {"TauJet", TauJet},
            {"ETM", ETM},
            {"ETT", ETT},
            {"HTT", HTT},
            {"HTM", HTM},
            {"JetCounts", JetCounts},
            {"HfBitCounts", HfBitCounts},
            {"HfRingEtSums", HfRingEtSums},
            {"TechTrig", TechTrig},
            {"Castor", Castor},
            {"BPTX", BPTX},
            {"GtExternal", GtExternal},
            {"ObjNull", ObjNull},
            {nullptr, (L1GtObject) - 1}
    };

    L1GtObject value = (L1GtObject) - 1;

    bool found = false;
    for (int i = 0; l1GtObjectStringToEnumMap[i].label && (!found); ++i)
        if (!std::strcmp(label.c_str(), l1GtObjectStringToEnumMap[i].label)) {
            found = true;
            value = l1GtObjectStringToEnumMap[i].value;
        }

    // in case of unrecognized L1GtObject, returns Mu
    // and write a warning (to not throw an exception)
    if (!found) {
        edm::LogInfo("L1GtObject") << "\n  '" << label
                << "' is not a recognized L1GtObject. \n  Return ObjNull.";

        value = ObjNull;
    }

    if (value == ObjNull) {
        edm::LogInfo("L1GtObject")
                << "\n  ObjNull means no valid L1GtObject defined!";
    }

    return value;
}

std::string l1GtObjectEnumToString(const L1GtObject& gtObject) {

    std::string gtObjectString;

    switch (gtObject) {

        case Mu: {
            gtObjectString = "Mu";
        }
            break;

        case NoIsoEG: {
            gtObjectString = "NoIsoEG";
        }
            break;

        case IsoEG: {
            gtObjectString = "IsoEG";
        }
            break;

        case CenJet: {
            gtObjectString = "CenJet";
        }
            break;

        case ForJet: {
            gtObjectString = "ForJet";
        }
            break;

        case TauJet: {
            gtObjectString = "TauJet";
        }
            break;

        case ETM: {
            gtObjectString = "ETM";
        }
            break;

        case ETT: {
            gtObjectString = "ETT";
        }
            break;

        case HTT: {
            gtObjectString = "HTT";
        }
            break;

        case HTM: {
            gtObjectString = "HTM";
        }
            break;

        case JetCounts: {
            gtObjectString = "JetCounts";
        }
            break;

        case HfBitCounts: {
            gtObjectString = "HfBitCounts";
        }
            break;

        case HfRingEtSums: {
            gtObjectString = "HfRingEtSums";
        }
            break;

        case TechTrig: {
            gtObjectString = "TechTrig";
        }
            break;

        case Castor: {
            gtObjectString = "Castor";
        }
            break;

        case BPTX: {
            gtObjectString = "BPTX";
        }
            break;

        case GtExternal: {
            gtObjectString = "GtExternal";
        }
            break;

        case ObjNull: {
            gtObjectString = "ObjNull";
            edm::LogInfo("L1GtObject")
                    << "\n  ObjNull means no valid L1GtObject defined!";
        }
            break;

        default: {
            edm::LogInfo("L1GtObject") << "\n  '" << gtObject
                    << "' is not a recognized L1GtObject. "
                    << "\n  Return ObjNull, which means no valid L1GtObject defined!";

            gtObjectString = "ObjNull";

        }
            break;
    }

    return gtObjectString;

}

