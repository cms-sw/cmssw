#ifndef CondFormats_L1TObjects_L1TGlobalParameters_h
#define CondFormats_L1TObjects_L1TGlobalParameters_h

/**
 * \class L1TGlobalParameters
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
class L1TGlobalParameters
{

public:

    // constructor
    L1TGlobalParameters();

    // destructor
    virtual ~L1TGlobalParameters();

public:

    /// get / set the number of bx in hardware
    inline  int gtTotalBxInEvent() const {
        return m_totalBxInEvent;
    }

    void setGtTotalBxInEvent(const int&);


    /// get / set the number of physics trigger algorithms
    inline unsigned int gtNumberPhysTriggers() const {
        return m_numberPhysTriggers;
    }

    void setGtNumberPhysTriggers(const unsigned int&);


    ///  get / set the number of L1 muons received by GT
    inline unsigned int gtNumberL1Mu() const {
        return m_numberL1Mu;
    }

    void setGtNumberL1Mu(const unsigned int&);

    ///  get / set the number of L1 e/gamma objects received by GT
    inline unsigned int gtNumberL1EG() const {
        return m_numberL1EG;
    }

    void setGtNumberL1EG(const unsigned int&);


    ///  get / set the number of L1  jets received by GT
    inline unsigned int gtNumberL1Jet() const {
        return m_numberL1Jet;
    }

    void setGtNumberL1Jet(const unsigned int&);


    ///  get / set the number of L1 tau  received by GT
    inline unsigned int gtNumberL1Tau() const {
        return m_numberL1Tau;
    }

    void setGtNumberL1Tau(const unsigned int&);



    /// hardware stuff

    ///   get / set the number of condition chips in GTL
    inline unsigned int gtNumberChips() const {
        return m_numberChips;
    }

    void setGtNumberChips(const unsigned int&);

    ///   get / set the number of pins on the GTL condition chips
    inline unsigned int gtPinsOnChip() const {
        return m_pinsOnChip;
    }

    void setGtPinsOnChip(const unsigned int&);

    ///   get / set the correspondence "condition chip - GTL algorithm word"
    ///   in the hardware
    inline const std::vector<int>& gtOrderOfChip() const {
        return m_orderOfChip;
    }

    void setGtOrderOfChip(const std::vector<int>&);

/*
    ///   get / set the number of PSB boards in GT
    inline int gtNumberPsbBoards() const {
        return m_numberPsbBoards;
    }

    void setGtNumberPsbBoards(const int&);

    ///   get / set the number of bits for eta of calorimeter objects
    inline unsigned int gtIfCaloEtaNumberBits() const {
        return m_ifCaloEtaNumberBits;
    }

    void setGtIfCaloEtaNumberBits(const unsigned int&);

    ///   get / set the number of bits for eta of muon objects
    inline unsigned int gtIfMuEtaNumberBits() const {
        return m_ifMuEtaNumberBits;
    }

    void setGtIfMuEtaNumberBits(const unsigned int&);

    ///    get / set WordLength
    inline int gtWordLength() const {
        return m_wordLength;
    }

    void setGtWordLength(const int&);

    ///    get / set one UnitLength
    inline int gtUnitLength() const {
        return m_unitLength;
    }

    void setGtUnitLength(const int&);
*/
    /// print all the L1 GT  parameters
    void print(std::ostream&) const;

private:

    /// bx in event
    int m_totalBxInEvent; 

    /// trigger decision

    /// number of physics trigger algorithms
    unsigned int m_numberPhysTriggers;

    /// trigger objects

    /// muons
    unsigned int m_numberL1Mu;

    /// e/gamma  objects
    unsigned int m_numberL1EG;


    ///  jets
    unsigned int m_numberL1Jet;
    
    ///  taus
    unsigned int m_numberL1Tau;



private:

    /// hardware

    /// number of condition chips
    unsigned int m_numberChips;

    /// number of pins on the GTL condition chips
    unsigned int m_pinsOnChip;

    /// correspondence "condition chip - GTL algorithm word" in the hardware
    std::vector<int> m_orderOfChip;
/*
    /// number of PSB boards in GT
    int m_numberPsbBoards;

    /// number of bits for eta of calorimeter objects
    unsigned int m_ifCaloEtaNumberBits;

    /// number of bits for eta of muon objects
    unsigned int m_ifMuEtaNumberBits;

private:

    /// GT DAQ record organized in words of WordLength bits
    int m_wordLength;

    /// one unit in the word is UnitLength bits
    int m_unitLength;
*/
};

#endif /*CondFormats_L1TObjects_L1TGlobalParameters_h*/
