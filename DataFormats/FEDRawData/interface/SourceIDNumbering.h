#ifndef DataFormats_FEDRawData_SourceIdNumbering_h
#define DataFormats_FEDRawData_SourceIdNumbering_h

/** \class SourceIDNumbering
 *
 *  This placeholdeer class will hold the fed numbering scheme for the CMS Source IDs
 *  for Phase-2 readout.  *  No two sources should have the same id. Each subdetector
 *  will have a reserved range. Gaps between ranges might give flexibility to the
 *  numbering. Total available range is unsigned 32-bit.
 *
 *  \author S. Morovic - UCSD
 */

class SourceIdNumbering {
public:
  enum {
    //dummy range definition for testing. This is not stable and will possible be reassigned
    MinDummySourceID = 0xfffffff0,
    MaxDummySourceID = 0xfffffffe,
    MAXSourceID = 0xffffffff
  };
};

#endif  // SourceIdNumbering_H
