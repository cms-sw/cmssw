#ifndef DTROMonitorFilter_H
#define DTROMonitorFilter_H

/** \class DTROMonitorFilter.h
 *  No description available.
 *
 *  $Date: 2012/01/21 14:56:54 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include <DataFormats/Common/interface/Handle.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>


class DTROMonitorFilter : public HLTFilter {
public:
  /// Constructor
  DTROMonitorFilter(const edm::ParameterSet&);

  /// Destructor
  virtual ~DTROMonitorFilter();

  // Operations
  virtual bool hltFilter(edm::Event& event, const edm::EventSetup& setup, trigger::TriggerFilterObjectWithRefs & filterproduct);
  
protected:

private:
  // Get the data integrity service
  edm::Handle<FEDRawDataCollection> rawdata;

  /// if not you need the label
  edm::InputTag inputLabel;

};
#endif

