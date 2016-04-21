#ifndef L1Trigger_L1TGlobal_MuCondition_h
#define L1Trigger_L1TGlobal_MuCondition_h

/**
 * \class MuCondition
 * 
 * 
 * Description: evaluation of a CondMuon condition.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 */

// system include files
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

// forward declarations
class GlobalCondition;
class MuonTemplate;

namespace l1t {

class L1MuGMTCand;

class GlobalBoard;

// class declaration
class MuCondition : public ConditionEvaluation
{

public:

    /// constructors
    ///     default
    MuCondition();

    ///     from base template condition (from event setup usually)
    MuCondition(const GlobalCondition*, const GlobalBoard*,
            const int nrL1Mu,
            const int ifMuEtaNumberBits);

    // copy constructor
    MuCondition(const MuCondition&);

    // destructor
    virtual ~MuCondition();

    // assign operator
    MuCondition& operator=(const MuCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const;

    /// print condition
    void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a Condition
    inline const MuonTemplate* gtMuonTemplate() const {
        return m_gtMuonTemplate;
    }

    void setGtMuonTemplate(const MuonTemplate*);

    ///   get / set the pointer to GTL
    inline const GlobalBoard* gtGTL() const {
        return m_gtGTL;
    }

    void setGtGTL(const GlobalBoard*);


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
    void copy(const MuCondition& cp);

    /// load muon candidates
    const l1t::Muon* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool checkObjectParameter(const int iCondition,
        const l1t::Muon& cand) const;

private:

    /// pointer to a MuonTemplate
    const MuonTemplate* m_gtMuonTemplate;

    /// pointer to GTL, to be able to get the trigger objects
    const GlobalBoard* m_gtGTL;

    /// number of bits for eta of muon objects
    int m_ifMuEtaNumberBits;

    // maximum number of bins for the delta phi scales
    unsigned int m_corrParDeltaPhiNrBins;



};

}
#endif
