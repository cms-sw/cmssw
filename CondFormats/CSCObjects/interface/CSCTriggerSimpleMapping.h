#ifndef CondFormats_CSCTriggerSimpleMapping_h
#define CondFormats_CSCTriggerSimpleMapping_h

/** 
 * \class CSCReadoutSimpleMapping
 * \author Lindsey Gray
 * A CSCTriggerMapping that encodes the hardware labels into a CSCDetId,
 * appropriate for most situations including slicetest.
 */

#include <CondFormats/CSCObjects/interface/CSCTriggerMapping.h>

class CSCTriggerSimpleMapping : public CSCTriggerMapping {
 public:

  /// Constructor
   CSCTriggerSimpleMapping();

  /// Destructor
   ~CSCTriggerSimpleMapping() override;

 private: 

   /**
     * Build a unique integer out of labels present or easily derivable from the 
     * readout.
     *
     */
    int hwId( int endcap, int station, int sector, int subsector, int cscid ) const override;

};

#endif
