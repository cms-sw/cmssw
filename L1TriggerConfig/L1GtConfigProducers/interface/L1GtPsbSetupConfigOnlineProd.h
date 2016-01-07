#ifndef L1GtConfigProducers_L1GtPsbSetupConfigOnlineProd_h
#define L1GtConfigProducers_L1GtPsbSetupConfigOnlineProd_h

/**
 * \class L1GtPsbSetupConfigOnlineProd
 *
 *
 * Description: online producer for L1GtPsbSetup.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files
#include "boost/shared_ptr.hpp"
#include <string>
#include <vector>

// user include files
//   base class
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"
#include "CondFormats/L1TObjects/interface/L1GtPsbSetup.h"
#include "CondFormats/DataRecord/interface/L1GtPsbSetupRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtPsbConfig.h"

// forward declarations

// class declaration
class L1GtPsbSetupConfigOnlineProd :
        public L1ConfigOnlineProdBase<L1GtPsbSetupRcd, L1GtPsbSetup>
{

public:

    /// constructor
    L1GtPsbSetupConfigOnlineProd(const edm::ParameterSet&);

    /// destructor
    ~L1GtPsbSetupConfigOnlineProd();

    /// public methods
    virtual boost::shared_ptr<L1GtPsbSetup> newObject(const std::string& objectKey);

private:

    /// A predicate to filter the column names of GT_SETUP for
    /// those that contain foreign keys to GT_PSB_SETUP.
    static bool notPsbColumnName(const std::string& columnName);

    /// Creates a new PSB object from a GT_PSB_SETUP entry and adds.
    void addPsbFromDb(const std::string& psbKey, std::vector<L1GtPsbConfig>& psbSetup);

    /// Creates a default valued PSB from an empty foreign key in the GT_SETUP table.
    void addDefaultPsb(const std::string& psbColumn, std::vector<L1GtPsbConfig>& psbSetup) const;

    /// Ensures that result contains exactly one line, returning false otherwise
    bool checkOneLineResult(
            const l1t::OMDSReader::QueryResults& result, const std::string& queryDescription) const;

    /// A wrapper for OMDSReader::QueryResults::fillVariable that throws an
    /// exception when the field it accesses was NULL.
    template<class T> void getRequiredValue(
            const l1t::OMDSReader::QueryResults& result, const std::string& colName, T& value) const {
        if (!result.fillVariable(colName, value)) {
            throw cms::Exception("NullValue") << "Required field " << colName
                    << " is NULL in database!";
        }
    }

    /// A function to extract a vector of booleans from the GT database format (NUMBER(1)
    /// columns labeled prefix<nn>suffix).
    std::vector<bool> extractBoolVector(
            const l1t::OMDSReader::QueryResults& query, const std::string& prefix,
            const std::string& suffix, unsigned nColumns) const;

    /// Concatenates prefix, number and suffix into a string.
    std::string numberedColumnName(
            const std::string& prefix, unsigned number, const std::string& suffix) const;

    /// Special case for empty suffix
    std::string numberedColumnName(const std::string& prefix, unsigned number) const;
    unsigned numberFromString(const std::string& aString) const;

};

#endif
