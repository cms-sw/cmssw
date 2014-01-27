#ifndef GlobalTrigger_L1uGtMuonCondition_h
#define GlobalTrigger_L1uGtMuonCondition_h

/**
 * \class L1uGtMuonCondition
 * 
 * 
 * Description: evaluation of a CondMuon condition.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna 
 * 
 *
 */

// system include files
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

// forward declarations
class L1GtCondition;
class L1GtMuonTemplate;

namespace l1t {

class L1MuGMTCand;

class L1uGtBoard;

// class declaration
class L1uGtMuonCondition : public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1uGtMuonCondition();

    ///     from base template condition (from event setup usually)
    L1uGtMuonCondition(const L1GtCondition*, const L1uGtBoard*,
            const int nrL1Mu,
            const int ifMuEtaNumberBits);

    // copy constructor
    L1uGtMuonCondition(const L1uGtMuonCondition&);

    // destructor
    virtual ~L1uGtMuonCondition();

    // assign operator
    L1uGtMuonCondition& operator=(const L1uGtMuonCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition() const;

    /// print condition
    void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1GtCondition
    inline const L1GtMuonTemplate* gtMuonTemplate() const {
        return m_gtMuonTemplate;
    }

    void setGtMuonTemplate(const L1GtMuonTemplate*);

    ///   get / set the pointer to GTL
    inline const L1uGtBoard* gtGTL() const {
        return m_gtGTL;
    }

    void setGtGTL(const L1uGtBoard*);


    ///   get / set the number of bits for eta of muon objects
    inline const int gtIfMuEtaNumberBits() const {
        return m_ifMuEtaNumberBits;
    }

    void setGtIfMuEtaNumberBits(const int&);


    ///   get / set maximum number of bins for the delta phi scales
    inline const int gtCorrParDeltaPhiNrBins() const {
        return m_corrParDeltaPhiNrBins;
    }

    void setGtCorrParDeltaPhiNrBins(const int&);


private:

    /// copy function for copy constructor and operator=
    void copy(const L1uGtMuonCondition& cp);

    /// load muon candidates
    const l1t::Muon* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool checkObjectParameter(const int iCondition,
        const l1t::Muon& cand) const;

private:

    /// pointer to a L1GtMuonTemplate
    const L1GtMuonTemplate* m_gtMuonTemplate;

    /// pointer to GTL, to be able to get the trigger objects
    const L1uGtBoard* m_gtGTL;

    /// number of bits for eta of muon objects
    int m_ifMuEtaNumberBits;

    // maximum number of bins for the delta phi scales
    unsigned int m_corrParDeltaPhiNrBins;



};

}
#endif
