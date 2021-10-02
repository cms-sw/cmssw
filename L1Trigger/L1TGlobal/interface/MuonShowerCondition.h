#ifndef L1Trigger_L1TGlobal_MuonShowerCondition_h
#define L1Trigger_L1TGlobal_MuonShowerCondition_h

/**
 * \class MuonShowerCondition
 *
 * Description: evaluation of a CondMuonShower condition.
 */

// system include files
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

#include "DataFormats/L1Trigger/interface/MuonShower.h"

// forward declarations
class GlobalCondition;
class MuonShowerTemplate;

namespace l1t {

  class GlobalBoard;

  // class declaration
  class MuonShowerCondition : public ConditionEvaluation {
  public:
    /// constructors
    ///     default
    MuonShowerCondition();

    ///     from base template condition (from event setup usually)
    MuonShowerCondition(const GlobalCondition*, const GlobalBoard*, const int nrL1MuShower);

    // copy constructor
    MuonShowerCondition(const MuonShowerCondition&);

    // destructor
    ~MuonShowerCondition() override;

    // assign operator
    MuonShowerCondition& operator=(const MuonShowerCondition&);

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const override;

    /// print condition
    void print(std::ostream& myCout) const override;

    ///   get / set the pointer to a Condition
    inline const MuonShowerTemplate* gtMuonShowerTemplate() const { return m_gtMuonShowerTemplate; }

    void setGtMuonShowerTemplate(const MuonShowerTemplate*);

    ///   get / set the pointer to GTL
    inline const GlobalBoard* gtGTL() const { return m_gtGTL; }

    void setGtGTL(const GlobalBoard*);

  private:
    /// copy function for copy constructor and operator=
    void copy(const MuonShowerCondition& cp);

    /// load muon candidates
    const l1t::MuonShower* getCandidate(const int bx, const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool checkObjectParameter(const int iCondition, const l1t::MuonShower& cand, const unsigned int index) const;

    /// pointer to a MuonShowerTemplate
    const MuonShowerTemplate* m_gtMuonShowerTemplate;

    /// pointer to GTL, to be able to get the trigger objects
    const GlobalBoard* m_gtGTL;
  };

}  // namespace l1t
#endif
