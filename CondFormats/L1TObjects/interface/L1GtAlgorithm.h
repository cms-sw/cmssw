#ifndef CondFormats_L1TObjects_L1GtAlgorithm_h
#define CondFormats_L1TObjects_L1GtAlgorithm_h

/**
 * \class L1GtAlgorithm
 *
 *
 * Description: L1 GT algorithm.
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

// system include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

// forward declarations

// class declaration
class L1GtAlgorithm
{

public:

    /// constructor(s)
    ///   empty
    L1GtAlgorithm();

    ///   name only
    L1GtAlgorithm(const std::string& algoNameValue);

    ///   name and logical expression
    L1GtAlgorithm(const std::string&, const std::string&);

    ///   name, logical expression and bit number
    L1GtAlgorithm(const std::string&, const std::string&, const int);

    /// destructor
    virtual ~L1GtAlgorithm();

public:

    /// get / set algorithm name
    inline const std::string algoName() const
    {
        return m_algoName;
    }

    inline void setAlgoName(const std::string& algoNameValue)
    {
        m_algoName = algoNameValue;
    }

    /// get / set algorithm alias
    inline std::string const & algoAlias() const
    {
        return m_algoAlias;
    }

    inline void setAlgoAlias(const std::string& algoAliasValue)
    {
        m_algoAlias = algoAliasValue;
    }

    /// get / set the logical expression for the algorithm
    inline std::string const & algoLogicalExpression() const
    {
        return m_algoLogicalExpression;
    }

    inline void setAlgoLogicalExpresssion(const std::string& logicalExpression)
    {
        m_algoLogicalExpression = logicalExpression;
    }

    /// return the RPN vector
    inline const std::vector<L1GtLogicParser::TokenRPN>& algoRpnVector() const {
        return m_algoRpnVector;
    }

    /// get / set algorithm bit number
    inline int algoBitNumber() const
    {
        return m_algoBitNumber;
    }

    inline void setAlgoBitNumber(const int algoBitNumberValue)
    {
        m_algoBitNumber = algoBitNumberValue;
    }

    /// get / set algorithm bit number
    inline const int algoChipNumber() const
    {
        return m_algoChipNumber;
    }

    inline void setAlgoChipNumber(const int algoChipNumberValue)
    {
        m_algoChipNumber = algoChipNumberValue;
    }


public:

    /// get the condition chip number the algorithm is located on
    const int algoChipNumber(const int numberConditionChips,
                         const int pinsOnConditionChip,
                         const std::vector<int>& orderConditionChip) const;

    /// get the output pin on the condition chip for the algorithm
    const int algoOutputPin(const int numberConditionChips,
                            const int pinsOnConditionChip,
                            const std::vector<int>& orderConditionChip) const;

    /// print condition
    virtual void print(std::ostream& myCout) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GtAlgorithm&);


private:

    /// algorithm name
    std::string m_algoName;

    /// algorithm alias
    std::string m_algoAlias;

    /// algorithm logical expression
    std::string m_algoLogicalExpression;

    /// algorithm RPN vector
    std::vector<L1GtLogicParser::TokenRPN> m_algoRpnVector;

    /// bit number (determined by output pin, chip number, chip order)
    /// the result for the algorithm is found at m_algoBitNumber position in
    /// the decision word vector<bool>
    int m_algoBitNumber;

    /// chip number (redundant with bit number)
    int m_algoChipNumber;


    COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtAlgorithm_h*/
