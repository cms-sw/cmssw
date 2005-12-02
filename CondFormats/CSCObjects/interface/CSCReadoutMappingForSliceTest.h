#ifndef CondFormats_CSCReadoutMappingForSliceTest_h
#define CondFormats_CSCReadoutMappingForSliceTest_h

/** 
 * \class CSCReadoutMappingForSliceTest
 * \author Tim Cox
 * A CSCReadoutMapping using encoding of hardware labels
 * appropriate for CSC Slice Test Winter 2005-6.
 */

#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>

class CSCReadoutMappingForSliceTest : public CSCReadoutMapping {
 public:

  /// Constructor
   CSCReadoutMappingForSliceTest();

  /// Destructor
   virtual ~CSCReadoutMappingForSliceTest();

 private: 

   /**
     * Build a unique integer out of the readout electronics labels.
     *
     * In general this must depend on endcap and station, as well as
     * vme crate number and dmb slot number. In principle perhaps tmb slot
     * number might not be neighbour of dmb?
     * But for slice test (Nov-2005) only relevant labels are vme and dmb.
     */
    int hwId( int endcap, int station, int vme, int dmb, int tmb ) const;

};

#endif
