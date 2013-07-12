#ifndef FEDRawData_FEDTrailer_h
#define FEDRawData_FEDTrailer_h

/** \class FEDTrailer
 *  TEMPORARY helper class to interpret/create FED trailer words.
 *  FIXME: will be replaced by the xdaq implementation
 *
 *  $Date: 2005/10/05 16:20:20 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

struct fedt_struct;

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
  const fedt_struct* theTrailer;

};
#endif

