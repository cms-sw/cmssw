#ifndef DTROMonitorFilter_H
#define DTROMonitorFilter_H

/** \class DTROMonitorFilter.h
 *  No description available.
 *
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
  virtual bool hltFilter(edm::Event& event, const edm::EventSetup& setup, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

protected:

private:
  /// if not you need the label
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;

};
#endif


/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
