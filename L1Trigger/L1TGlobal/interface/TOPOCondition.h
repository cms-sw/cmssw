#ifndef L1Trigger_L1TGlobal_TOPOCondition_h
#define L1Trigger_L1TGlobal_TOPOCondition_h

/**
 * \class TOPOCondition
 *
 * Description: evaluation of a CondTOPO condition.
 */

#include <iosfwd>
#include <string>
#include <utility>

#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

#include "hls4ml/emulator.h"

// forward declarations
class GlobalCondition;
class TOPOTemplate;

namespace l1t {

  class GlobalBoard;

  // class declaration
  class TOPOCondition : public ConditionEvaluation {
  public:
    /// constructors
    ///     default
    TOPOCondition();

    ///     from base template condition (from event setup usually)
    TOPOCondition(const GlobalCondition*, const GlobalBoard*);

    // copy constructor
    TOPOCondition(const TOPOCondition&);
    // destructor
    ~TOPOCondition() override;

    // assign operator
    TOPOCondition& operator=(const TOPOCondition&);

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const override;

    /// print condition
    void print(std::ostream& myCout) const override;

    ///   get / set the pointer to a Condition
    inline const TOPOTemplate* gtTOPOTemplate() const { return m_gtTOPOTemplate; }

    void setGtTOPOTemplate(const TOPOTemplate*);

    ///   get / set the pointer to GTL
    inline const GlobalBoard* gtGTB() const { return m_gtGTB; }

    void setuGtB(const GlobalBoard*);

    /// get/set score value
    void setScore(const float scoreval) const;

    inline float getScore() const { return m_savedscore; }

    void loadModel();

    inline hls4mlEmulator::ModelLoader const& model_loader() const { return m_model_loader; }

  private:
    /// copy function for copy constructor and operator=
    void copy(const TOPOCondition& cp);

    /// pointer to a TOPOTemplate
    const TOPOTemplate* m_gtTOPOTemplate;

    /// pointer to uGt GlobalBoard, to be able to get the trigger objects
    const GlobalBoard* m_gtGTB;

    static constexpr char const* kModelNamePrefix = "topo_";

    hls4mlEmulator::ModelLoader m_model_loader;
    std::shared_ptr<hls4mlEmulator::Model> m_model;

    ///axo score for possible score saving
    mutable float m_savedscore;
  };

}  // namespace l1t
#endif
