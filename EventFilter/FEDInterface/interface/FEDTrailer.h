#ifndef EventFilter_FEDInterface_FEDTrailer_h
#define EventFilter_FEDInterface_FEDTrailer_h

/** \class FEDTrailer
 *  TEMPORARY helper class to interpret/create FED trailer words.
 *  FIXME: will be replaced by the xdaq implementation
 *
 *  \author N. Amapane - CERN
 */

#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/fed_trailer.h"


class FEDTrailer {
public:
  /// Constructor
  FEDTrailer(const unsigned char* trailer);

  /// Destructor
  virtual ~FEDTrailer();
  
  /// The length of the event fragment counted in 64-bit words including header and trailer
  int lenght();

  /// Cyclic Redundancy Code of the event fragment including header and trailer
  int crc();

  /// Event fragment status information
  int evtStatus();

  /// Current value of the Trigger Throttling System bitsAQ).
  int ttsBits();
 
  /// 0 -> the current trailer word is the last one.
  /// 1-> other trailer words can follow 
  /// (always 0 for ECAL)
  bool moreTrailers();

  // Check that the trailer is OK
  bool check();

  /// Set all fields in the trailer
  static void set(unsigned char* trailer,
		  int evt_lgth,
		  int crc,  
		  int evt_stat,
		  int tts,
		  bool T=false);

private:
  const fedt_t* theTrailer;

};
#endif

