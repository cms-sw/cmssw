#ifndef CondFormats_CSCTriggerElectronicsMapping_h
#define CondFormats_CSCTriggerElectronicsMapping_h

/** 
 * \class CSCReadoutElectronicsMapping
 * \author Lindsey Gray
 * A CSCTriggerMapping that encodes the eletronics labels into a unique label,
 * appropriate for most situations including slicetest.
 */

#include <CondFormats/CSCObjects/interface/CSCTriggerMapping.h>

class CSCTriggerElectronicsMapping : public CSCTriggerMapping {
 public:

  /// Constructor
   CSCTriggerElectronicsMapping();

  /// Destructor
   ~CSCTriggerElectronicsMapping() override;

 private: 

   /**
     * Build a unique integer out of labels present in or easily derivable from the 
     * readout.
     *
     */
    int hwId( int SPboardId, int FPGA, int cscid, int zero1=0, int zero2=0 ) const override;

};

#endif
