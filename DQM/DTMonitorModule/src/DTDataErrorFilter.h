#ifndef DTDataErrorFilter_H
#define DTDataErrorFilter_H

/** \class DTDataErrorFilter
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
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
  virtual bool filter(edm::Event& event, const edm::EventSetup& setup);
  
protected:

private:
  DTDataIntegrityTask * dataMonitor;

  
};
#endif

