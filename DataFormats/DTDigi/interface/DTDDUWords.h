#ifndef DTDigi_DTDDUWords_h
#define DTDigi_DTDDUWords_h

/** \file
 * MASKS and SHIFTS definition. Documentation at:
 *
 * https://uimon.cern.ch/twiki/bin/view/CMS/FEDDataFormats
 */

#define WORDCONTROLMASK 0xE0000000
#define WORDCONTROLSHIFT 29
#define WORDTYPEMASK 0x1F000000
#define WORDTYPESHIFT 24

/// to distinguish between ROS and TDC error
#define ERRORMASK 0x8000
#define ERRORSHIFT 15

#define DEBUG_TYPE_MASK 0xE00000
#define DEBUG_TYPE_SHIFT 21
#define DEBUG_MESSAGE_MASK 0x7FFF
#define CEROS_ID_CEROS_STATUS_MASK 0x1F0000
#define CEROS_ID_CEROS_STATUS_SHIFT 16
#define EV_ID_CEROS_STATUS_MASK 0xFC0
#define EV_ID_CEROS_STATUS_SHIFT 6
#define DONTREAD_CEROS_STATUS_MASK 0x3F
#define CEROS_ID_ROS_STATUS_MASK 0x3F


#define TTC_EVENT_COUNTER_MASK 0xFFFFFF

#define TFF_MASK 0x800000
#define TFF_SHIFT 23
#define TPX_MASK 0x400000
#define TPX_SHIFT 22
#define L1A_FIFO_OCC_MASK 0x3F0000
#define L1A_FIFO_OCC_SHIFT 16
#define EVENT_WORD_COUNT_MASK 0xFFFF

#define ERROR_TYPE_MASK 0xE00000
#define ERROR_TYPE_SHIFT 21
#define ERROR_ROB_ID_MASK 0x1F0000
#define ERROR_ROB_ID_SHIFT 16
#define ERROR_CEROS_ID_MASK 0x3F

#define ROB_ID_MASK 0x1F000000 
#define EVENT_ID_MASK 0xFFF000
#define EVENT_ID_SHIFT 12
#define BUNCH_ID_MASK 0xFFF
#define WORD_COUNT_MASK 0xFFF

#define PC_MASK 0x8000000
#define PC_SHIFT 27
#define PAF_MASK 0x4000000
#define PAF_SHIFT 26
#define TDC_ID_MASK 0x3000000
#define TDC_ID_SHIFT 24

#define TDC_CHANNEL_MASK 0xF80000
#define TDC_CHANNEL_SHIFT 19
#define TDC_TIME_MASK 0x7FFFC // First two bits are excluded
#define TDC_TIME_SHIFT 2

#define TDC_ERROR_MASK 0x7FFF

#define SCFO_MASK 0xFF

#define TRIGGER_WORD_COUNT_MASK 0xFFFF

#define TRIGGER_DATA_MASK 0xFFFF


#define SC_LAT_SHIFT 8
#define SC_LAT_MASK 0x7F

#define SC_NW_MASK 0xFF

#define SC_TRIGGERDLY_MASK 0x7
#define SC_TRIGGERDLY_SHIFT 12
#define SC_BXC_MASK 0xFFF



#include <boost/cstdint.hpp>
#include <iostream>



/** \class DTROSWordType
 *  Enumeration of DT Read Out Sector (ROS) word types.
 *
 * \author M. Zanetti - INFN Padova
 */
class DTROSWordType {

public:
  
  /// Constructor
  DTROSWordType(const uint32_t index) {
    word_ = index;
  }
  
  DTROSWordType(const DTROSWordType& obj) {
    *this = obj;
  }

  DTROSWordType() : word_(0) {};
  
  /// Destructor
  virtual ~DTROSWordType() {};

  /// List of DT DDU Word Types
  enum wordTypes {
    ROSHeader = 1,
    ROSTrailer = 2,
    ROSError = 3,
    GroupHeader = 4,
    GroupTrailer = 5,
    TDCHeader = 6,
    TDCTrailer = 7,
    TDCMeasurement = 8,
    TDCError = 9,
    SCHeader = 10,
    SCTrailer = 11,
    SCData = 12,
    ROSDebug = 13,
    TDCDebug = 14,
    Control = 15
  };


  /// DDU word type getter 
  enum wordTypes type() {
    
    enum wordTypes wordType = Control;
    
    // ROS/ROB/SC Headers
    if ( ((word_ & WORDCONTROLMASK) >> WORDCONTROLSHIFT) == headerControlWord ) {
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == rosTypeWord ) wordType = ROSHeader;
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == scTypeWord ) wordType = SCHeader;
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) < scTypeWord ) wordType = GroupHeader;
    }

    // ROS/ROB/SC Trailers
    if ( ((word_ & WORDCONTROLMASK) >> WORDCONTROLSHIFT) == trailerControlWord ) {
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == rosTypeWord ) wordType = ROSTrailer;
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == scTypeWord ) wordType = SCTrailer;
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) < scTypeWord ) wordType = GroupTrailer;
    }

    // TDC Header
    if ( ((word_ & WORDCONTROLMASK) >> WORDCONTROLSHIFT) == tdcHeaderControlWord ) 
      wordType = TDCHeader;

    // TDC Trailer
    if ( ((word_ & WORDCONTROLMASK) >> WORDCONTROLSHIFT) == tdcTrailerControlWord ) 
      wordType = TDCTrailer;

    // TDC/SC Data
    if ( ((word_ & WORDCONTROLMASK) >> WORDCONTROLSHIFT) == tdcDataControlWord ) {
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == scTypeWord ) wordType = SCData;
      else wordType = TDCMeasurement;
    }

    // Errors
    if ( ((word_ & WORDCONTROLMASK) >> WORDCONTROLSHIFT) == errorControlWord ) {
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == rosTypeWord ) wordType = ROSError;
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) < scTypeWord ) wordType = TDCError;
    }

    // ROS Debug
    if ( ((word_ & WORDCONTROLMASK) >> WORDCONTROLSHIFT) == debugControlWord ) {
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == rosTypeWord ) wordType = ROSDebug;
      if ( ((word_ & WORDTYPEMASK) >> WORDTYPESHIFT) < scTypeWord ) wordType = TDCDebug;
    }


    return wordType;
  }


  /// Control bits definitions
  static const uint32_t headerControlWord = 0;
  static const uint32_t trailerControlWord = 1;
  static const uint32_t tdcHeaderControlWord = 2;
  static const uint32_t tdcTrailerControlWord = 3;
  static const uint32_t tdcDataControlWord = 4;
  static const uint32_t errorControlWord = 6;
  static const uint32_t debugControlWord = 7;

  /// Word Type bits definitions
  static const uint32_t rosTypeWord = 31;
  static const uint32_t scTypeWord = 25;


private:

   uint32_t word_;
  
};



/** \class DTROSHeaderWord
 *  DT ROS Header interpreter. 
 *  It interprets the TTC Event counter (24 bits).
 *
 * \author M. Zanetti - INFN Padova
 */
class DTROSHeaderWord {

public:

  /// Constructor
  DTROSHeaderWord() {}

  DTROSHeaderWord(const DTROSHeaderWord& obj) { *this = obj; }

  DTROSHeaderWord(const uint32_t index) : 
    word_(index) {} 

  /// Destructor
  virtual ~DTROSHeaderWord() {}

  int TTCEventCounter() const { return  word_ & TTC_EVENT_COUNTER_MASK; }

  static void set(uint32_t &word,
		  int ttc_event_counter) {
    
    word = 
      DTROSWordType::headerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord << WORDTYPESHIFT |
      ttc_event_counter;
  }


private:

  uint32_t word_;

};


/** \class DTROSTrailerWord
 *  DT ROS Trailer interpreter. 
 *  Information interpreted: 
 *  - TFF: L1 FIFO is full (1 bit)
 *  - TPX: Transmitter parity (1 bit)
 *  - L1A FIFO occupancy  (6 bits)
 *  - Event Word count (16 bits)
 *
 * \author M. Zanetti - INFN Padova
 */
class DTROSTrailerWord {

public:
  
  /// Constructor
  DTROSTrailerWord() {}

  DTROSTrailerWord(const DTROSTrailerWord& obj) { *this = obj; }

  DTROSTrailerWord(const uint32_t index) : 
    word_(index) {} 

  /// Destructor
  virtual ~DTROSTrailerWord() {}

  int TFF() const { return (word_ & TFF_MASK) >> TFF_SHIFT; }
  int TPX() const { return (word_ & TPX_MASK) >> TPX_SHIFT; }
  int l1AFifoOccupancy() const { return (word_ & L1A_FIFO_OCC_MASK) >> L1A_FIFO_OCC_SHIFT; }
  int EventWordCount() const { return word_ & EVENT_WORD_COUNT_MASK; }

  static void set(uint32_t &word,
		  int tff,
		  int tpx,
		  int l1a_fifo_occ,
		  int event_word_count) {
    
    word = 
      DTROSWordType::trailerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord << WORDTYPESHIFT |
      tff << TFF_SHIFT |
      tpx << TPX_SHIFT |
      l1a_fifo_occ << L1A_FIFO_OCC_SHIFT |
      event_word_count;
  }


private:

  uint32_t word_;

};


/** \class DTROSErrorWord
 *  DT ROS Error interpreter. 
 *  It interprets the Error type, the ROB_ID (2 bits) and the CEROS ID (6 bits)
 *
 * \author M. Zanetti - INFN Padova
 */
class DTROSErrorWord {

public:

  /// Constructor
  DTROSErrorWord() {}

  DTROSErrorWord(const DTROSErrorWord& obj) { *this = obj; }

  DTROSErrorWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTROSErrorWord() {}

  int errorType() const { return (word_ & ERROR_TYPE_MASK) >> ERROR_TYPE_SHIFT;} 
  int robID() const { return (word_ & ERROR_ROB_ID_MASK) >> ERROR_ROB_ID_SHIFT;}
  int cerosID() const {return errorType()==4 ? (word_ & ERROR_CEROS_ID_MASK) : 0;}

  static void set(uint32_t &word,
		  int error_type,
		  int rob_id) {
    
    word = 
      DTROSWordType::errorControlWord << WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord << WORDTYPESHIFT |
      error_type << ERROR_TYPE_SHIFT |
      rob_id << ERROR_ROB_ID_SHIFT |
      1 << ERRORSHIFT;
  }

private:

  uint32_t word_;

};


/** \class DTROSDebugWord
 *  DT ROS Debug interpreter. 
 *  It interprets the Debug type (3 bits) and the debug message 
 *  (in the first 15 bits) 
 *
 * \author M. Zanetti - INFN Padova
 */
class DTROSDebugWord {

public:

  /// Constructor
  DTROSDebugWord() {}

  DTROSDebugWord(const DTROSDebugWord& obj) { *this = obj; }

  DTROSDebugWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTROSDebugWord() {}

  int debugType() const { return (word_ & DEBUG_TYPE_MASK) >> DEBUG_TYPE_SHIFT;} 
  int debugMessage() const { return (word_ & DEBUG_MESSAGE_MASK) ;} 
  int cerosIdCerosStatus() const { return debugType()==3 ? (word_ & CEROS_ID_CEROS_STATUS_MASK) >> CEROS_ID_CEROS_STATUS_SHIFT : 0;} 
  int evIdMis() const { return debugType()==3 ? (word_ & EV_ID_CEROS_STATUS_MASK) >> EV_ID_CEROS_STATUS_SHIFT : 0;} 
  int dontRead() const { return debugType()==3 ? (word_ & DONTREAD_CEROS_STATUS_MASK) : 0;} 
  int cerosIdRosStatus() const { return debugType()==4 ? (word_ & CEROS_ID_ROS_STATUS_MASK) : 0;} 

  static void set(uint32_t &word,                                 //CB FIXME do we need setters for DEBUG Types 3 and 4? 
		  int debug_type) {
    
    word = 
      DTROSWordType::debugControlWord << WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord << WORDTYPESHIFT |
      debug_type << DEBUG_TYPE_SHIFT |
      504 << 15; // TEMPORARY
  }

  static void set(uint32_t &word,                                 
		  int debug_type,
		  int ceros_id) {
    
    word = 
      DTROSWordType::debugControlWord << WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord << WORDTYPESHIFT |
      debug_type << DEBUG_TYPE_SHIFT |
      ceros_id << CEROS_ID_CEROS_STATUS_SHIFT |
      1 << 15;
  }

private:

  uint32_t word_;

};


/** \class DTROBHeaderWord
 *  DT ROB Header interpreter. 
 *  It interprets the ROB_ID (5 bits), the Event ID (12 bits) 
 *  and the Bunch ID (12 bits).
 *
 * \author M. Zanetti - INFN Padova
 */
class DTROBHeaderWord {

public:

  /// Constructor
  DTROBHeaderWord() {}

  DTROBHeaderWord(const DTROBHeaderWord& obj) { *this = obj; }

  DTROBHeaderWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTROBHeaderWord() {}

  int robID() const { return (word_ & ROB_ID_MASK) >> WORDTYPESHIFT;} 
  int eventID() const { return (word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;} 
  int bunchID() const { return (word_ & BUNCH_ID_MASK);} 


  static void set(uint32_t &word,
		  int rob_id,
		  int event_id,
		  int bunch_id) {
    
    word = 
      DTROSWordType::headerControlWord << WORDCONTROLSHIFT |
      rob_id << WORDTYPESHIFT |
      event_id << EVENT_ID_SHIFT |
      bunch_id;
  }
  

private:

  uint32_t word_;

};


/** \class DTROBTrailerWord
 *  DT ROB Trailer interpreter. 
 *  It interprets the ROB_ID (5 bits), the Event ID (12 bits) 
 *  and the Word ID (12 bits).
 *
 * \author M. Zanetti - INFN Padova
 */
class DTROBTrailerWord {

public:

  /// Constructor
  DTROBTrailerWord() {}

  DTROBTrailerWord(const DTROBTrailerWord& obj) { *this = obj; }

  DTROBTrailerWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTROBTrailerWord() {}

  int robID() const { return (word_ & ROB_ID_MASK) >> WORDTYPESHIFT;} 
  int eventID() const { return (word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;} 
  int wordCount() const { return (word_ & WORD_COUNT_MASK);} 

  static void set(uint32_t &word,
		  int rob_id,
		  int event_id,
		  int word_count) {
    
    word = 
      DTROSWordType::trailerControlWord << WORDCONTROLSHIFT |
      rob_id << WORDTYPESHIFT |
      event_id << EVENT_ID_SHIFT |
      word_count;
  }
  

private:

  uint32_t word_;
};




/** \class DTTDCHeaderWord
 *  DT TDC Header interpreter. 
 *  It interprets the Parity Checks, FIFO occupancy, Lokeced channels (all 1 bit),
 *  the TDC_ID (2 bits), the Event ID (12 bits) and the Bunch ID (12 bits).
 *
 * \author M. Zanetti - INFN Padova
 */
class DTTDCHeaderWord {

public:

  /// Constructor
  DTTDCHeaderWord() {}

  DTTDCHeaderWord(const DTTDCHeaderWord& obj) { *this = obj; }

  DTTDCHeaderWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTTDCHeaderWord() {}

  int PC() const { return (word_ & PC_MASK) >> PC_SHIFT;} 
  int PAF() const { return (word_ & PAF_MASK) >> PAF_SHIFT;} 
  int HU() const { return (word_ & PAF_MASK) >> PAF_SHIFT;} /// <== OBSOLETE!!
  int tdcID() const { return (word_ & TDC_ID_MASK) >> TDC_ID_SHIFT;} 
  int eventID() const { return (word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;} 
  int bunchID() const { return (word_ & BUNCH_ID_MASK);} 

  static void set(uint32_t &word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int event_id,
		  int bunch_id) {
    
    word = 
      DTROSWordType::tdcHeaderControlWord << WORDCONTROLSHIFT |
      pc << PC_SHIFT |
      paf << PAF_SHIFT |
      hu << PAF_SHIFT |
      tdc_id << TDC_ID_SHIFT |
      event_id << EVENT_ID_SHIFT |
      bunch_id;
  }


private:

  uint32_t word_;
};


/** \class DTTDCTrailerWord
 *  DT TDC Trailer interpreter. 
 *  It interprets the Parity Checks, FIFO occupancy, Lokeced channels (all 1 bit),
 *  the TDC_ID (2 bits), the Event ID (12 bits) and the Word ID (12 bits).
 *
 * \author M. Zanetti - INFN Padova
 */
class DTTDCTrailerWord {

public:

  /// Constructor
  DTTDCTrailerWord() {}

  DTTDCTrailerWord(const DTTDCTrailerWord& obj) { *this = obj; }

  DTTDCTrailerWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTTDCTrailerWord() {}

  int PC() const { return (word_ & PC_MASK) >> PC_SHIFT;} 
  int PAF() const { return (word_ & PAF_MASK) >> PAF_SHIFT;} 
  int HU() const { return (word_ & PAF_MASK) >> PAF_SHIFT;} /// <== OBSOLETE!!
  int tdcID() const { return (word_ & TDC_ID_MASK) >> TDC_ID_SHIFT;} 
  int eventID() const { return (word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;} 
  int wordCount() const { return (word_ & WORD_COUNT_MASK);} 

  static void set(uint32_t &word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int event_id,
		  int word_count) {
    
    word = 
      DTROSWordType::tdcTrailerControlWord << WORDCONTROLSHIFT |
      pc << PC_SHIFT |
      paf << PAF_SHIFT |
      hu << PAF_SHIFT |
      tdc_id << TDC_ID_SHIFT |
      event_id << EVENT_ID_SHIFT |
      word_count;
  }

private:

  uint32_t word_;
};


/** \class DTTDCMeasurementWord
 *  DT TDC Measurement interpreter. 
 *  It interprets the Parity Checks, FIFO occupancy, Lokeced channels (all 1 bit),
 *  the TDC_ID (2 bits), the TDC channel (5 bits), and the TDC time (19 bits)
 *
 * \author M. Zanetti - INFN Padova
 */
class DTTDCMeasurementWord {

public:

  /// Constructor
  DTTDCMeasurementWord() {}
  
  DTTDCMeasurementWord(const DTTDCMeasurementWord& obj) { *this = obj; }

  DTTDCMeasurementWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTTDCMeasurementWord() {}

  int PC() const { return (word_ & PC_MASK) >> PC_SHIFT;} 
  int PAF() const { return (word_ & PAF_MASK) >> PAF_SHIFT;} 
  int HU() const { return (word_ & PAF_MASK) >> PAF_SHIFT;} /// <== OBSOLETE!!
  int tdcID() const { return (word_ & TDC_ID_MASK) >> TDC_ID_SHIFT;} 
  int tdcChannel() const { return (word_ & TDC_CHANNEL_MASK) >> TDC_CHANNEL_SHIFT;} 
  int tdcTime() const { return (word_ & TDC_TIME_MASK) >> TDC_TIME_SHIFT;} 


  static void set(uint32_t &word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int tdc_channel,
		  int tdc_time) {
    
    word = 
      DTROSWordType::tdcDataControlWord << WORDCONTROLSHIFT |
      pc << PC_SHIFT |
      paf << PAF_SHIFT |
      hu << PAF_SHIFT |
      tdc_id << TDC_ID_SHIFT |
      tdc_channel << TDC_CHANNEL_SHIFT |
      tdc_time;
  }



private:

  uint32_t word_;
};


/** \class DTTDCErrorWord
 *  DT TDC Error interpreter. 
 *  It interprets the Parity Checks, FIFO occupancy, Lokeced channels (all 1 bit),
 *  the TDC_ID (2 bits) and the TDC error flag (15 bits)
 *
 * \author M. Zanetti - INFN Padova
 */
class DTTDCErrorWord {

public:

  /// Constructor
  DTTDCErrorWord() {}
  
  DTTDCErrorWord(const DTTDCErrorWord& obj) { *this = obj; }

  DTTDCErrorWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTTDCErrorWord() {}

  int PC() const { return (word_ & PC_MASK) >> PC_SHIFT;} 
  int PAF() const { return (word_ & PAF_MASK) >> PAF_SHIFT;} 
  int HU() const { return (word_ & PAF_MASK) >> PAF_SHIFT;} /// <== OBSOLETE!!
  int tdcID() const { return (word_ & TDC_ID_MASK) >> TDC_ID_SHIFT;} 
  int tdcError() const { return (word_ & TDC_ERROR_MASK);} 

  static void set(uint32_t &word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int tdc_error) {
    
    word = 
      DTROSWordType::errorControlWord << WORDCONTROLSHIFT |
      pc << PC_SHIFT |
      paf << PAF_SHIFT |
      hu << PAF_SHIFT |
      tdc_id << TDC_ID_SHIFT |
      0 << ERRORSHIFT |
      tdc_error;
  }

private:

  uint32_t word_;
};


/** \class DTLocalTriggerHeaderWord
 *  DT Sector Collector header interpreter. 
 *  It interprets ROS event ID (12 bits) and the Sector Collector FIFO occupancy (8 bits)
 *
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerHeaderWord {

public:

  /// Constructor
  DTLocalTriggerHeaderWord() {}
  
  DTLocalTriggerHeaderWord(const DTLocalTriggerHeaderWord& obj) { *this = obj; }

  DTLocalTriggerHeaderWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTLocalTriggerHeaderWord() {}

  int eventID() const { return (word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;}
  int SCFO() const { return (word_ & SCFO_MASK);}

  
  static void set(uint32_t &word,
		  int event_id,
		  int scfo) {
    
    word = 
      DTROSWordType::headerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord << WORDTYPESHIFT |
      event_id << EVENT_ID_SHIFT |
      scfo; 
  }

private:

  uint32_t word_;
};


/** \class DTLocalTriggerTrailerWord
 *  DT Sector Collector trailer interpreter. 
 *  It interprets the word count (16 bits)
 *
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerTrailerWord {

public:

  /// Constructor
  DTLocalTriggerTrailerWord() {}
  
  DTLocalTriggerTrailerWord(const DTLocalTriggerTrailerWord& obj) { *this = obj; }

  DTLocalTriggerTrailerWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTLocalTriggerTrailerWord() {}

  int wordCount() const { return (word_ & TRIGGER_WORD_COUNT_MASK);}

  static void set(uint32_t &word,
		  int word_count) {
    
    word = 
      DTROSWordType::trailerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord << WORDTYPESHIFT |
      word_count; 
  }


private:

  uint32_t word_;
};


/** \class DTLocalTriggerDataWord
 *  DT Sector Collector data interpreter. 
 *  It interprets the Sector Collector data (16 bits)
 *
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerDataWord {

public:

  /// Constructor
  DTLocalTriggerDataWord() {}
  
  DTLocalTriggerDataWord(const DTLocalTriggerDataWord& obj) { *this = obj; }
 
  DTLocalTriggerDataWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTLocalTriggerDataWord() {}

  int SCData() const { return (word_ & TRIGGER_DATA_MASK);}

  int getBits(int first) const { 
    return first==1 ? ((word_ & TRIGGER_DATA_MASK) >> 8) : ((word_ & TRIGGER_DATA_MASK)&0xFF);  
  }

  //int hasTrigger(int first) const { return (getBits(first) & 0x40) >> 6; }
  int hasTrigger(int first) const { return (trackQuality(first) != 7? 1 : 0);}
  int trackQuality(int first) const { return (getBits(first) & 0xE) >> 1; }
 

  static void set(uint32_t &word,
		  int sc_data) {
    
    word = 
      DTROSWordType::tdcDataControlWord << WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord << WORDTYPESHIFT |
      sc_data; 
  }


private:

  uint32_t word_;
};


/** \class DTDDUFirstStatusWord
 *  DT DDU status 1 interpreter (8 bits word). 
 *  It interprets the error messages from each DDU channel
 *
 * \author M. Zanetti - INFN Padova
 */
class DTDDUFirstStatusWord {

public:

  /// Constructor
  DTDDUFirstStatusWord() {}

  DTDDUFirstStatusWord(const DTDDUFirstStatusWord& obj) { *this = obj; }

  DTDDUFirstStatusWord(const unsigned char index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTDDUFirstStatusWord() {}

  int channelEnabled() const { return (word_ & 0x1);}
  int timeout() const { return (word_ & 0x2) >> 1;}
  int eventTrailerLost() const { return (word_ & 0x4) >> 2;}
  int opticalFiberSignalLost() const { return (word_ & 0x8) >> 3;}
  int tlkPropagationError() const { return (word_ & 0x10) >> 4;}
  int tlkPatternError() const { return (word_ & 0x20) >> 5;}
  int tlkSignalLost() const { return (word_ & 0x40) >> 6;}
  int errorFromROS() const { return (word_ & 0x80) >> 7;}


private:

  unsigned char word_;
};


/** \class DTDDUSecondStatusWord
 *  DT DDU status 2 interpreter. 
 *  It interprets the (16 bits)
 *  WARNING!! : It interprets the second part of a 64 bits word!
 *
 * \author M. Zanetti - INFN Padova
 */
class DTDDUSecondStatusWord {
  
public:
  
  /// Constructor
  DTDDUSecondStatusWord() {}

  DTDDUSecondStatusWord(const DTDDUSecondStatusWord& obj) { *this = obj; }

  DTDDUSecondStatusWord(const uint32_t index) : 
    word_(index) {}
  
  /// Destructor
  virtual ~DTDDUSecondStatusWord() {}
  
  int l1AIDError() const { return (word_ & 0x1); }
  int bxIDError() const { return (word_ & 0x2) >> 1; }
  int fifoFull() const { return (word_ & 0x1C ) >> 2; }
  int inputFifoFull() const { return (word_ & 0xE0) >> 5; }
  int fifoAlmostFull() const { return (word_ & 0x700) >> 8; }
  int inputFifoAlmostFull() const { return (word_ & 0x3800) >> 11; }
  int outputFifoFull() const { return (word_ & 0x4000) >> 14; }
  int outputFifoAlmostFull() const { return (word_ & 0x8000) >> 15; }
  int rosList() const {return (word_ & 0xFFF0000) >> 16; }
  int warningROSPAF() const {return (word_ & 0x10000000) >> 28; }
  int busyROSPAF() const {return (word_ & 0x20000000) >> 29; }
  int outOfSynchROSError() const {return (word_ & 0x40000000) >> 30; }
  

//   int fifoAlmostEmpty() const { return (word_ & 0x1C000) >> 14; }
//   int inputFifoAlmostEmpty() const { return (word_ & 0xE0000) >> 17; }
//   int outputFifoFull() const { return (word_ & 0x100000) >> 20; }
//   int outputFifoAlmostFull() const { return (word_ & 0x200000) >> 21; }
//   int outputFifoAlmostEmpty() const { return (word_ & 0x400000) >> 22; }

private:
  
  uint32_t word_;
};


/** \class DTLocalTriggerSectorCollectorHeaderWord
 *  DT Sector Collector private header interpreter. 
 *  It interprets Latency measured by SC-Latency Timer Unit (still testing!)
 *  and the number of 16-bit words following this header sent by the Sector Collector
 *
 * \author R. Travaglini - INFN Bologna
 */
class DTLocalTriggerSectorCollectorHeaderWord {

public:

  /// Constructor
  DTLocalTriggerSectorCollectorHeaderWord() {}
  
  DTLocalTriggerSectorCollectorHeaderWord(const DTLocalTriggerSectorCollectorHeaderWord& obj) { *this = obj; }

  DTLocalTriggerSectorCollectorHeaderWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTLocalTriggerSectorCollectorHeaderWord() {}

  int Latency() const { return ((word_  >> SC_LAT_SHIFT) &  SC_LAT_MASK);}
  int NumberOf16bitWords() const { return (word_ & SC_NW_MASK);}

  
  static void set(uint32_t &word,
		  int lat,
		  int nw) {
    
    word = 
      DTROSWordType::headerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord << WORDTYPESHIFT |
      (lat & SC_LAT_MASK) << SC_LAT_SHIFT |
      (nw & SC_NW_MASK) ; 
  }

private:

  uint32_t word_;
};


/** \class DTLocalTriggerSectorCollectorSubHeaderWord
 *  DT Sector Collector private SUB-header interpreter. 
 *  It interprets local SC bunch Counter and delay (3-bit) between trigger used to stop spying and
 *  effective bx stop
 *
 * \author R. Travaglini - INFN Bologna
 */
class DTLocalTriggerSectorCollectorSubHeaderWord {

public:

  /// Constructor
  DTLocalTriggerSectorCollectorSubHeaderWord() {}
  
  DTLocalTriggerSectorCollectorSubHeaderWord(const DTLocalTriggerSectorCollectorSubHeaderWord& obj) { *this = obj; }

  DTLocalTriggerSectorCollectorSubHeaderWord(const uint32_t index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTLocalTriggerSectorCollectorSubHeaderWord() {}

  int TriggerDelay() const { return ((word_  >>SC_TRIGGERDLY_SHIFT ) & SC_TRIGGERDLY_MASK);}
  int LocalBunchCounter() const { return (word_ & SC_BXC_MASK );}

  
  static void set(uint32_t &word,
		  int trigdly,
		  int bxcount) {
    
    word = 
      DTROSWordType::headerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord << WORDTYPESHIFT |
      (trigdly &  SC_TRIGGERDLY_MASK) << SC_TRIGGERDLY_SHIFT |
      (bxcount & SC_BXC_MASK) ; 
  }

private:

  uint32_t word_;
};


#endif
