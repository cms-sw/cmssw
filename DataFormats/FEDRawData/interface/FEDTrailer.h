#ifndef DataFormats_FEDRawData_FEDTrailer_h
#define DataFormats_FEDRawData_FEDTrailer_h

/** \class FEDTrailer
 *  Helper class to interpret/create FED trailer words.
 *
 *  \author N. Amapane - CERN, R. Mommsen - FNAL
 */

#include <cstdint>

struct fedt_struct;

class FEDTrailer {
public:
  /// Constructor
  FEDTrailer(const unsigned char* trailer);

  /// Destructor
  virtual ~FEDTrailer();

  /// The length of the event fragment counted in 64-bit words including header and trailer
  uint32_t fragmentLength() const;

  /// Cyclic Redundancy Code of the event fragment including header and trailer
  uint16_t crc() const;

  /// Event fragment status information
  uint8_t evtStatus() const;

  /// Current value of the Trigger Throttling System bits
  uint8_t ttsBits() const;

  /// 0 -> the current trailer word is the last one.
  /// 1 -> other trailer words can follow
  bool moreTrailers() const;

  /// True if the CRC value has been modified by the S-link sender card
  bool crcModified() const;

  /// True if the FRL has detected a transmission error over the s-link cable
  bool slinkError() const;

  /// True if the FED_ID given by the FED is not the one expected by the FRL
  bool wrongFedId() const;

  /// Check that the trailer is OK
  bool check() const;

  /// Return the word containing the consistency checks
  uint32_t conscheck() const;

  /// Set all fields in the trailer
  static void set(unsigned char* trailer,
		  uint32_t lenght,
		  uint16_t crc,
		  uint8_t evt_stat,
		  uint8_t tts,
		  bool moreTrailers=false);

  static const uint32_t length;

private:
  const fedt_struct* theTrailer;

};
#endif // DataFormats_FEDRawData_FEDTrailer_h
