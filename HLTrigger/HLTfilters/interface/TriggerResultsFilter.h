#ifndef TriggerResultsFilter_h
#define TriggerResultsFilter_h

/** \class TriggerResultsFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  arbitrary logical combinations of L1 and HLT results.
 *
 *  It has been written as an extension of the HLTHighLevel and HLTHighLevelDev 
 *  filters.
 *
 *  $Date: 2010/02/18 14:45:06 $
 *  $Revision: 1.8 $
 *
 *  Authors: Martin Grunewald, Andrea Bocci
 *
 */

#include <vector>
#include <string>

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"

// forward declaration
namespace triggerExpression {
  class Evaluator;
}

//
// class declaration
//

class TriggerResultsFilter : public HLTFilter {
public:
  explicit TriggerResultsFilter(const edm::ParameterSet &);
  ~TriggerResultsFilter();
  virtual bool filter(edm::Event &, const edm::EventSetup &);

private:
  /// parse the logical expression into functionals
  void parse(const std::string & expression);

  /// evaluator for the trigger condition
  triggerExpression::Evaluator * m_expression;

  /// cache some data from the Event for faster access by the m_expression
  triggerExpression::Data m_eventCache;
};

#endif //TriggerResultsFilter_h
