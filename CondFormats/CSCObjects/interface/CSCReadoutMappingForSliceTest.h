#ifndef CondFormats_CSCReadoutMappingForSliceTest_h
#define CondFormats_CSCReadoutMappingForSliceTest_h

/** 
 * \class CSCReadoutMappingForSliceTest
 * \author Tim Cox
 * A CSCReadoutMapping using encoding of hardware labels
 * appropriate for CSC Slice Test from Winter 2005 to Summer 2006 (at least).
 */

#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>

class CSCReadoutMappingForSliceTest : public CSCReadoutMapping {
public:
  /// Constructor
  CSCReadoutMappingForSliceTest();

  /// Destructor
  ~CSCReadoutMappingForSliceTest() override;

private:
  /**
     * Build a unique integer out of the readout electronics labels.
     *
     * In general this must depend on endcap and station, as well as
     * vme crate number and dmb slot number. In principle perhaps tmb slot
     * number might not be neighbour of dmb?
     * But for slice test (Nov-2005 on) only relevant labels are vme and dmb.
     */
  int hwId(int endcap, int station, int vme, int dmb, int tmb) const override;
};

#endif
