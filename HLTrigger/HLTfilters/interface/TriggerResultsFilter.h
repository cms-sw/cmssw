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
 *  $Date: 2010/01/17 15:29:46 $
 *  $Revision: 1.1 $
 *
 *  Authors: Martin Grunewald, Andrea Bocci
 *
 */

#include <vector>
#include <string>

#include "FWCore/Framework/interface/TriggerNames.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/HLTfilters/interface/TriggerExpressionData.h"

// forward declaration
namespace hlt {
  class TriggerExpressionEvaluator;
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
  /// HLT TriggerResults EDProduct
  edm::InputTag m_triggerResults;

  /// throw on any requested trigger being unknown
  bool m_throw;

  /// list of required HLT triggers by HLT index
  std::vector<hlt::TriggerExpressionEvaluator *> m_expressions;

  /// cache some data from the Event for faster access by the m_expressions
  hlt::TriggerExpressionData m_eventCache;
};

#endif //TriggerResultsFilter_h
