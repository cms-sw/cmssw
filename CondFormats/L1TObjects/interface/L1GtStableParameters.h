#ifndef CondFormats_L1TObjects_L1GtStableParameters_h
#define CondFormats_L1TObjects_L1GtStableParameters_h

/**
 * \class L1GtStableParameters
 * 
 * 
 * Description: L1 GT stable parameters.  
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
#include <vector>

#include <ostream>

#include <boost/cstdint.hpp>

// user include files
//   base class

// forward declarations

// class declaration
class L1GtStableParameters
{

public:

    // constructor
    L1GtStableParameters();

    // destructor
    virtual ~L1GtStableParameters();

public:

    /// get / set the number of physics trigger algorithms
    inline unsigned int gtNumberPhysTriggers() const
    {
        return m_numberPhysTriggers;
    }

    void setGtNumberPhysTriggers(const unsigned int&);

    /// get / set the additional number of physics trigger algorithms
    inline unsigned int gtNumberPhysTriggersExtended() const
    {
        return m_numberPhysTriggersExtended;
    }

    void setGtNumberPhysTriggersExtended(const unsigned int&);

    /// get / set the number of technical triggers
    inline unsigned int gtNumberTechnicalTriggers() const
    {
        return m_numberTechnicalTriggers;
    }

    void setGtNumberTechnicalTriggers(const unsigned int&);

    ///  get / set the number of L1 muons received by GT
    inline unsigned int gtNumberL1Muons() const
    {
        return m_numberL1Muons;
    }

    void setGtNumberL1Muons(const unsigned int&);


    ///  get / set the number of L1 e/gamma objects received by GT
    inline unsigned int gtNumberL1EGamma() const
    {
        return m_numberL1EGamma;
    }

    void setGtNumberL1EGamma(const unsigned int&);


    ///  get / set the number of L1 isolated e/gamma objects received by GT
    inline unsigned int gtNumberL1IsolatedEGamma() const
    {
        return m_numberL1IsolatedEGamma;
    }

    void setGtNumberL1IsolatedEGamma(const unsigned int&);

    ///  get / set the number of L1 central jets received by GT
    inline unsigned int gtNumberL1CentralJets() const
    {
        return m_numberL1CentralJets;
    }

    void setGtNumberL1CentralJets(const unsigned int&);

    ///  get / set the number of L1 forward jets received by GT
    inline unsigned int gtNumberL1ForwardJets() const
    {
        return m_numberL1ForwardJets;
    }

    void setGtNumberL1ForwardJets(const unsigned int&);

    ///  get / set the number of L1 tau jets received by GT
    inline unsigned int gtNumberL1TauJets() const
    {
        return m_numberL1TauJets;
    }

    void setGtNumberL1TauJets(const unsigned int&);

    ///  get / set the number of L1 jet counts received by GT
    inline unsigned int gtNumberL1JetCounts() const
    {
        return m_numberL1JetCounts;
    }

    void setGtNumberL1JetCounts(const unsigned int&);


    /// hardware stuff

    ///   get / set the number of condition chips in GTL
    inline unsigned int gtNumberConditionChips() const
    {
        return m_numberConditionChips;
    }

    void setGtNumberConditionChips(const unsigned int&);

    ///   get / set the number of pins on the GTL condition chips
    inline unsigned int gtPinsOnConditionChip() const
    {
        return m_pinsOnConditionChip;
    }

    void setGtPinsOnConditionChip(const unsigned int&);


    ///   get / set the correspondence "condition chip - GTL algorithm word"
    ///   in the hardware
    inline std::vector<int> gtOrderConditionChip() const
    {
        return m_orderConditionChip;
    }

    void setGtOrderConditionChip(const std::vector<int>&);

    ///   get / set the number of PSB boards in GT
    inline int gtNumberPsbBoards() const
    {
        return m_numberPsbBoards;
    }

    void setGtNumberPsbBoards(const int&);

    ///    get / set WordLength
    inline int gtWordLength() const
    {
        return m_wordLength;
    }

    void setGtWordLength(const int&);

    ///    get / set one UnitLength
    inline int gtUnitLength() const
    {
        return m_unitLength;
    }

    void setGtUnitLength(const int&);


    /// print all the L1 GT stable parameters
    void print(std::ostream&) const;

private:

    /// trigger decision

    /// number of physics trigger algorithms
    unsigned int m_numberPhysTriggers;

    /// additional number of physics trigger algorithms
    unsigned int m_numberPhysTriggersExtended;

    /// number of technical triggers
    unsigned int m_numberTechnicalTriggers;

    /// trigger objects

    /// muons
    unsigned int m_numberL1Muons;

    /// e/gamma and isolated e/gamma objects
    unsigned int m_numberL1EGamma;
    unsigned int m_numberL1IsolatedEGamma;

    /// central, forward and tau jets
    unsigned int m_numberL1CentralJets;
    unsigned int m_numberL1ForwardJets;
    unsigned int m_numberL1TauJets;

    /// jet counts
    unsigned int m_numberL1JetCounts;

private:

    /// hardware

    /// number of condition chips
    unsigned int m_numberConditionChips;

    /// number of pins on the GTL condition chips
    unsigned int m_pinsOnConditionChip;

    /// correspondence "condition chip - GTL algorithm word" in the hardware
    /// chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    std::vector<int> m_orderConditionChip;

    /// number of PSB boards in GT
    int m_numberPsbBoards;

private:

    /// GT DAQ record organized in words of WordLength bits
    int m_wordLength;

    /// one unit in the word is UnitLength bits
    int m_unitLength;

};

#endif /*CondFormats_L1TObjects_L1GtStableParameters_h*/
