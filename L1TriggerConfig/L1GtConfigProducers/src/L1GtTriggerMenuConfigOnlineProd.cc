/**
 * \class L1GtTriggerMenuConfigOnlineProd
 *
 *
 * Description: online producer for L1GtTriggerMenu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuConfigOnlineProd.h"

// system include files
#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtTriggerMenuConfigOnlineProd::L1GtTriggerMenuConfigOnlineProd(const edm::ParameterSet& parSet) :
    L1ConfigOnlineProdBase<L1GtTriggerMenuRcd, L1GtTriggerMenu> (parSet) {

    // empty

}

// destructor
L1GtTriggerMenuConfigOnlineProd::~L1GtTriggerMenuConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtTriggerMenu> L1GtTriggerMenuConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // shared pointer for L1GtTriggerMenu
    boost::shared_ptr<L1GtTriggerMenu> pL1GtTriggerMenu = boost::shared_ptr<L1GtTriggerMenu>(
            new L1GtTriggerMenu());

    // l1GtTriggerMenu: parameters in table GTFE_SETUP_FK

    const std::string gtSchema = "CMS_GT";

    // SQL query:
    //   SELECT SCALES_FK FROM CMS_GT.L1T_MENU WHERE L1T_MENU.ID = objectKey

    l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
            "SCALES_FK", gtSchema, "L1T_MENU", "L1T_MENU.ID", m_omdsReader.singleAttribute(
                    objectKey));

    // check if query was successful
    if (results.queryFailed()) {
        edm::LogError("L1-O2O") << "Problem with L1GtTriggerMenu key:" << objectKey;
        return pL1GtTriggerMenu;
    }

    std::string scalesKey;
    results.fillVariable("SCALES_FK", scalesKey);


    // fill the record

    // trigger menu name - "L1T_MENU.ID"
    pL1GtTriggerMenu->setGtTriggerMenuName(objectKey);


    if (edm::isDebugEnabled()) {
        std::ostringstream myCoutStream;
        int verbosity = 2;
        pL1GtTriggerMenu->print(myCoutStream, verbosity);
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\nThe following L1GtTriggerMenu record was read from OMDS: \n"
                << myCoutStream.str() << "\n" << std::endl;

        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\nPrimary key for L1 scales: \n"
                << scalesKey << "\n" << std::endl;

    }

    return pL1GtTriggerMenu;
}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtTriggerMenuConfigOnlineProd);
