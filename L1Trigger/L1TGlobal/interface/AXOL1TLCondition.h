#ifndef L1Trigger_L1TGlobal_AXOL1TLCondition_h
#define L1Trigger_L1TGlobal_AXOL1TLCondition_h

/**
 * \class AXOL1TLCondition
 *
 * Description: evaluation of a CondAXOL1TL condition.
 */

// system include files
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

// forward declarations
class GlobalCondition;
class AXOL1TLTemplate;

namespace l1t {

  class L1Candidate;
  class GlobalBoard;

  // class declaration
  class AXOL1TLCondition : public ConditionEvaluation {
  public:
    /// constructors
    ///     default
    AXOL1TLCondition();

    ///     from base template condition (from event setup usually)
    AXOL1TLCondition(const GlobalCondition*, const GlobalBoard*);

    // copy constructor
    AXOL1TLCondition(const AXOL1TLCondition&);
    // destructor
    ~AXOL1TLCondition() override;

    // assign operator
    AXOL1TLCondition& operator=(const AXOL1TLCondition&);

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const override;

    /// print condition
    void print(std::ostream& myCout) const override;

    ///   get / set the pointer to a Condition
    inline const AXOL1TLTemplate* gtAXOL1TLTemplate() const { return m_gtAXOL1TLTemplate; }

    void setGtAXOL1TLTemplate(const AXOL1TLTemplate*);

    ///   get / set the pointer to GTL
    inline const GlobalBoard* gtGTB() const { return m_gtGTB; }

    void setuGtB(const GlobalBoard*);

    //get / set AXOL1TL model version
    inline const std::string gtModelVerion() const { return m_AXOL1TLmodelversion; }

    void setModelVersion(const std::string modelversionname);

  private:
    /// copy function for copy constructor and operator=
    void copy(const AXOL1TLCondition& cp);

    /// pointer to a AXOL1TLTemplate
    const AXOL1TLTemplate* m_gtAXOL1TLTemplate;

    /// pointer to uGt GlobalBoard, to be able to get the trigger objects
    const GlobalBoard* m_gtGTB;

    //to set modelversion from globalboard<-globalproducer<-config
    std::string m_AXOL1TLmodelversion = "NONE";
  };

}  // namespace l1t
#endif
