#ifndef special_HLTDTROMonitorFilter_H
#define special_HLTDTROMonitorFilter_H

/** \class HLTDTROMonitorFilter.h
 *  No description available.
 *
 *  $Date: 2009/05/20 16:12:45 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include <DataFormats/Common/interface/Handle.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>


class HLTDTROMonitorFilter : public HLTFilter {
public:
  /// Constructor
  HLTDTROMonitorFilter(const edm::ParameterSet&);

  /// Destructor
  virtual ~HLTDTROMonitorFilter();

  // Operations
  virtual bool filter(edm::Event& event, const edm::EventSetup& setup);
  
protected:

private:
  // Get the data integrity service
  edm::Handle<FEDRawDataCollection> rawdata;

  /// if not you need the label
  edm::InputTag inputLabel;

};
#endif

