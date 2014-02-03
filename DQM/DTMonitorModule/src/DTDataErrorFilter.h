#ifndef DTDataErrorFilter_H
#define DTDataErrorFilter_H

/** \class DTDataErrorFilter
 *  No description available.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class DTDataIntegrityTask;


class DTDataErrorFilter : public HLTFilter {
public:
  /// Constructor
  DTDataErrorFilter(const edm::ParameterSet&);

  /// Destructor
  virtual ~DTDataErrorFilter();

  // Operations
  virtual bool hltFilter(edm::Event& event, const edm::EventSetup& setup, trigger::TriggerFilterObjectWithRefs & filterproduct);
  
protected:

private:
  DTDataIntegrityTask * dataMonitor;

  
};
#endif

