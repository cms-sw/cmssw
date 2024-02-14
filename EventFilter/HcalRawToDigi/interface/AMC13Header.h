/* -*- C++ -*- */
#ifndef AMC13Header_H_included
#define AMC13Header_H_included

#include <cstdint>

namespace hcal {
  /** \class AMC13Header
   *
   *  Interpretive class for the AMC13XG common data format
   */
  class AMC13Header {
  public:
    /** get the source id from the CDF header */
    inline int sourceId() const { return int(cdfHeader >> 8) & 0xFFF; }
    /** get the bunch id from the CDF header */
    inline int bunchId() const { return int(cdfHeader >> 20) & 0xFFF; }
    /** get the Event Number from the CDF header */
    inline int l1aNumber() const { return int((cdfHeader >> 32) & 0x00FFFFFF); }
    /** Get the Event Type value (2007.11.03 - Not defined, but should stay consistent among events.) */
    inline int CDFEventType() const { return (int(cdfHeader >> 56) & 0x0F); }
    /** Get the Orbit Number from the CDF. */
    inline unsigned int orbitNumber() const { return (unsigned int)((amc13Header >> 4) & 0xFFFFFFFFu); }
    /** Get the number of modules in the payload */
    inline int NAMC() const { return int((amc13Header >> 52) & 0xF); }
    /** Get the format version number */
    inline int AMC13FormatVersion() const { return int((amc13Header >> 60) & 0xF); }

    // Per-AMC items
    /** Get the board identifier for the given module (sequential) */
    inline uint16_t AMCId(int i) const { return uint16_t(modulesHeaders[i] & 0xFFFF); }
    /** Get the slot for the given module (sequential) */
    inline int AMCSlot(int i) const { return int((modulesHeaders[i] >> 16) & 0xF); }
    /** Get the block number */
    inline int AMCBlockNumber(int i) const { return int((modulesHeaders[i] >> 20) & 0xFF); }
    /** Get the size */
    inline int AMCSize(int i) const { return int((modulesHeaders[i] >> 32) & 0xFFFFFF); }
    /** More blocks? */
    inline bool AMCMore(int i) const { return ((modulesHeaders[i] >> 61) & 0x1) != 0; }
    /** Segmented data? */
    inline bool AMCSegmented(int i) const { return ((modulesHeaders[i] >> 60) & 0x1) != 0; }
    /** Was the length as expected? (logic appears inverted in firmware) */
    inline bool AMCLengthOk(int i) const { return ((modulesHeaders[i] >> 62) & 0x1) == 0; }
    /** Was the CRC correct as received by the AMC13? */
    inline bool AMCCRCOk(int i) const { return ((modulesHeaders[i] >> 56) & 0x1) != 0; }
    /** Is there data for this AMC? */
    inline bool AMCDataPresent(int i) const { return ((modulesHeaders[i] >> 58) & 0x1) != 0; }
    /** Does the EvN and BCN match for this AMC? */
    inline bool AMCDataValid(int i) const { return ((modulesHeaders[i] >> 57) & 0x1) != 0; }
    /** Is this AMC input enabled? */
    inline bool AMCEnabled(int i) const { return ((modulesHeaders[i] >> 59) & 0x1) != 0; }

    /** Get the pointer to the beginning of the data for the given AMC */
    const uint64_t* AMCPayload(int i) const;

  private:
    uint64_t cdfHeader;
    uint64_t amc13Header;
    uint64_t modulesHeaders[12];
  };
}  // namespace hcal

#endif  // AMC13Header_H_included
