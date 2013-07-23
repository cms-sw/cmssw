#ifndef special_HLTDTROMonitorFilter_H
#define special_HLTDTROMonitorFilter_H

/** \class HLTDTROMonitorFilter.h
 *  No description available.
 *
 *  $Date: 2012/01/21 15:00:13 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class HLTDTROMonitorFilter : public edm::EDFilter {
public:
  /// Constructor
  HLTDTROMonitorFilter(const edm::ParameterSet&);

  /// Destructor
  virtual ~HLTDTROMonitorFilter();

  // Operations
  virtual bool filter(edm::Event& event, const edm::EventSetup& setup);
  
protected:

private:
  edm::InputTag inputLabel;
};
#endif

