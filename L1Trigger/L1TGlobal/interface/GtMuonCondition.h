#ifndef GlobalTrigger_GtMuonCondition_h
#define GlobalTrigger_GtMuonCondition_h

/**
 * \class GtMuonCondition
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
class GlobalMuonTemplate;

namespace l1t {

class L1MuGMTCand;

class GtBoard;

// class declaration
class GtMuonCondition : public ConditionEvaluation
{

public:

    /// constructors
    ///     default
    GtMuonCondition();

    ///     from base template condition (from event setup usually)
    GtMuonCondition(const GlobalCondition*, const GtBoard*,
            const int nrL1Mu,
            const int ifMuEtaNumberBits);

    // copy constructor
    GtMuonCondition(const GtMuonCondition&);

    // destructor
    virtual ~GtMuonCondition();

    // assign operator
    GtMuonCondition& operator=(const GtMuonCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const;

    /// print condition
    void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a Condition
    inline const GlobalMuonTemplate* gtMuonTemplate() const {
        return m_gtMuonTemplate;
    }

    void setGtMuonTemplate(const GlobalMuonTemplate*);

    ///   get / set the pointer to GTL
    inline const GtBoard* gtGTL() const {
        return m_gtGTL;
    }

    void setGtGTL(const GtBoard*);


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
    void copy(const GtMuonCondition& cp);

    /// load muon candidates
    const l1t::Muon* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool checkObjectParameter(const int iCondition,
        const l1t::Muon& cand) const;

private:

    /// pointer to a GlobalMuonTemplate
    const GlobalMuonTemplate* m_gtMuonTemplate;

    /// pointer to GTL, to be able to get the trigger objects
    const GtBoard* m_gtGTL;

    /// number of bits for eta of muon objects
    int m_ifMuEtaNumberBits;

    // maximum number of bins for the delta phi scales
    unsigned int m_corrParDeltaPhiNrBins;



};

}
#endif
