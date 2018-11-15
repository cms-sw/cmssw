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
	    {"ETMHF", gtETMHF},
	    {"TowerCount",gtTowerCount},
	    {"MinBiasHFP0", gtMinBiasHFP0},
	    {"MinBiasHFM0", gtMinBiasHFM0},
	    {"MinBiasHFP1", gtMinBiasHFP1},
	    {"MinBiasHFM1", gtMinBiasHFM1},
	    {"ETTem", gtETTem},
	    {"AsymEt", gtAsymmetryEt},
	    {"AsymHt", gtAsymmetryHt},
	    {"AsymEtHF", gtAsymmetryEtHF},
	    {"AsymEtHF", gtAsymmetryHtHF},
	    {"CENT0", gtCentrality0},
	    {"CENT1", gtCentrality1},
	    {"CENT2", gtCentrality2},
	    {"CENT3", gtCentrality3},
	    {"CENT4", gtCentrality4},
	    {"CENT5", gtCentrality5},
	    {"CENT6", gtCentrality6},
	    {"CENT7", gtCentrality7},
            {"External", gtExternal},
            {"ObjNull", ObjNull},
            {nullptr, (GlobalObject) - 1}
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

        case gtETMHF: {
            gtObjectString = "ETMHF";
        }
            break;

        case gtTowerCount: {
            gtObjectString = "TowerCount";
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

        case gtETTem: {
            gtObjectString = "ETTem";
        }
            break;

        case gtAsymmetryEt: {
            gtObjectString = "AsymEt";
        }
            break;

        case gtAsymmetryHt: {
            gtObjectString = "AsymHt";
        }
            break;

        case gtAsymmetryEtHF: {
            gtObjectString = "AsymEtHF";
        }
            break;

        case gtAsymmetryHtHF: {
            gtObjectString = "AsymHtHF";
        }
            break;

        case gtCentrality0: {
            gtObjectString = "CENT0";
        }
            break;

        case gtCentrality1: {
            gtObjectString = "CENT1";
        }
            break;

        case gtCentrality2: {
            gtObjectString = "CENT2";
        }
            break;

        case gtCentrality3: {
            gtObjectString = "CENT3";
        }
            break;

        case gtCentrality4: {
            gtObjectString = "CENT4";
        }
            break;

        case gtCentrality5: {
            gtObjectString = "CENT5";
        }
            break;

        case gtCentrality6: {
            gtObjectString = "CENT6";
        }
            break;

        case gtCentrality7: {
            gtObjectString = "CENT7";
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
