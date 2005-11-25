#ifndef DTRawToDigi_DTDDUWords_h
#define DTRawToDigi_DTDDUWords_h

/** \file
 * MASKS and SHIFTS definition. Documentation at:
 *
 * https://uimon.cern.ch/twiki/bin/view/CMS/FEDDataFormats
 */

#define WORDTYPEMASK 0xFF000000
#define WORDTYPESHIFT 24
#define WORDCONTROLSHIFT 29

/// to distinguish between ROS and TDC error
#define ERRORMASK 0x8000
#define ERRORSHIFT 15


#define TTC_EVENT_COUNTER_MASK 0xFFFFFF

#define TFF_MASK 0x800000
#define TFF_SHIFT 23
#define TPX_MASK 0x400000
#define TPX_SHIFT 22
#define ECHO_MASK 0x300000
#define ECHO_SHIFT 20
#define ECLO_MASK 0xC0000
#define ECLO_SHIFT 18
#define BCO_MASK 0x30000
#define BCO_SHIFT 16
#define EVENT_WORD_COUNT_MASK 0xFFFF

#define ERROR_TYPE_MASK 0xE00000
#define ERROR_TYPE_SHIFT 21
#define ERROR_ROB_ID_MASK 0x1F0000
#define ERROR_ROB_ID_SHIFT 16

#define ROB_ID_MASK 0x1F000000 
#define EVENT_ID_MASK 0xFFF000
#define EVENT_ID_SHIFT 12
#define BUNCH_ID_MASK 0xFFF
#define WORD_COUNT_MASK 0xFFF

#define PC_MASK 0x10000000
#define PC_SHIFT 28
#define PAF_MASK 0x8000000
#define PAF_SHIFT 27
#define HU_MASK 0x4000000
#define HU_SHIFT 26
#define TDC_ID_MASK 0x3000000
#define TDC_ID_SHIFT 24

#define TDC_CHANNEL_MASK 0xF80000
#define TDC_CHANNEL_SHIFT 19
#define TDC_TIME_MASK 0x7FFFF

#define TDC_ERROR_MASK 0x7FFF

#define SCFO_MASK 0xFF

#define TRIGGER_WORD_COUNT_MASK 0xFFFF

#define TRIGGER_DATA_MASK 0xFFFF


// To be removed
#define DTDDU_WORD_SIZE 4
#define SLINK_WORD_SIZE 8


/** \class DTROSWordType
 *  Enumeration of DT Read Out Sector (ROS) word types.
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTROSWordType {

public:
  
  /// Constructor
  DTROSWordType(const unsigned int* index) : 
    word_(index) {};

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
    Control = 13
  };


  /// DDU word type getter 
  enum wordTypes type() {
    
    enum wordTypes wordType = Control;
    
    // ROS/ROB/SC Headers
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == headerControlWord ) {
      if ( ((*word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == rosTypeWord ) wordType = ROSHeader;
      if ( ((*word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == scTypeWord ) wordType = SCHeader;
      if ( ((*word_ & WORDTYPEMASK) >> WORDTYPESHIFT) < scTypeWord ) wordType = GroupHeader;
    }

    // ROS/ROB/SC Trailers
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == trailerControlWord ) {
      if ( ((*word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == rosTypeWord ) wordType = ROSTrailer;
      if ( ((*word_ & WORDTYPEMASK) >> WORDTYPESHIFT) == scTypeWord ) wordType = SCTrailer;
      if ( ((*word_ & WORDTYPEMASK) >> WORDTYPESHIFT) < scTypeWord ) wordType = GroupTrailer;
    }

    // TDC Header
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == tdcHeaderControlWord ) 
      wordType = TDCHeader;

    // TDC Trailer
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == tdcTrailerControlWord ) 
      wordType = TDCTrailer;

    // TDC Measurement
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == tdcDataControlWord ) 
      wordType = TDCMeasurement;

    // Errors
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == errorControlWord ) {
      if ( (*word_ & ERRORMASK) == 1 ) wordType = ROSError;
      if ( (*word_ & ERRORMASK) == 0 ) wordType = TDCError;
    }

    /// FIXME >>>>
    // No Implementation so far for Trigger data!!!! (ambiguos..)
    // No Implementation for Debugging data too
    /// >>>>>>>>>>

    return wordType;
  }


  /// Update the word by a ROS word size ( == 32 bits) 
  void update(const unsigned int* index) {
    word_ = index; 
  }


  /// Control bits definitions
  static const unsigned int headerControlWord = 0;
  static const unsigned int trailerControlWord = 1;
  static const unsigned int tdcHeaderControlWord = 2;
  static const unsigned int tdcTrailerControlWord = 3;
  static const unsigned int tdcDataControlWord = 4;
  static const unsigned int errorControlWord = 6;

  /// Word Type bits definitions
  static const unsigned int rosTypeWord = 31;
  static const unsigned int scTypeWord = 25;


private:

  const unsigned int * word_;
  
};



/** \class DTROSHeaderWord
 *  DT ROS Header interpreter. 
 *  It interprets the TTC Event counter (24 bits).
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTROSHeaderWord {

public:

  /// Constructor
  DTROSHeaderWord(const unsigned int* index) : 
    word_(index) {} 

  /// Destructor
  virtual ~DTROSHeaderWord() {}

  int TTCEventCounter() { return  *word_ & TTC_EVENT_COUNTER_MASK; }

  static void set(unsigned char* word,
		  int ttc_event_counter) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::headerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord << WORDTYPESHIFT |
      ttc_event_counter;
  }


private:

  const unsigned int * word_;

};


/** \class DTROSTrailerWord
 *  DT ROS Trailer interpreter. 
 *  Information interpreted: 
 *  - TFF: L1 FIFO is full (1 bit)
 *  - TPX: Transmitter parity (1 bit)
 *  - ECHO: Event Counter High FIFO occupancy (2 bits)
 *  - ECLO: Event COunter Low FIFO occupancy (2 bits)
 *  - BCO: Bunch Counter FIFO occupancy (2 bits)
 *  - Event Word count (16 bits)
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTROSTrailerWord {

public:
  
  /// Constructor
  DTROSTrailerWord(const unsigned int* index) : 
    word_(index) {} 

  /// Destructor
  virtual ~DTROSTrailerWord() {}

  int TFF() { return (*word_ & TFF_MASK) >> TFF_SHIFT; }
  int TPX() { return (*word_ & TPX_MASK) >> TPX_SHIFT; }
  int ECHO() { return (*word_ & ECHO_MASK) >> ECHO_SHIFT; }
  int ECLO() { return (*word_ & ECLO_MASK) >> ECLO_SHIFT; }
  int BCO() { return (*word_ & BCO_MASK) >> BCO_SHIFT; }
  int EventWordCount() { return *word_ & EVENT_WORD_COUNT_MASK; }

  static void set(unsigned char* word,
		  int tff,
		  int tpx,
		  int echo,
		  int eclo,
		  int bco,
		  int event_word_count) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::trailerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord << WORDTYPESHIFT |
      tff << TFF_SHIFT |
      tpx << TPX_SHIFT |
      echo << ECHO_SHIFT |
      eclo << ECLO_SHIFT |
      bco << BCO_SHIFT |
      event_word_count;
  }


private:

  const unsigned int * word_;

};


/** \class DTROSErrorWord
 *  DT ROS Error interpreter. 
 *  It interprets the Error type and the ROB_ID (2 bits) 
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTROSErrorWord {

public:

  /// Constructor
  DTROSErrorWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTROSErrorWord() {}

  int errorType() { return (*word_ & ERROR_TYPE_MASK) >> ERROR_TYPE_SHIFT;} 
  int robID() { return (*word_ & ERROR_ROB_ID_MASK) >> ERROR_ROB_ID_SHIFT;} 

  static void set(unsigned char* word,
		  int error_type,
		  int rob_id) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::errorControlWord << WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord << WORDTYPESHIFT |
      error_type << ERROR_TYPE_SHIFT |
      rob_id << ERROR_ROB_ID_SHIFT |
      1 << ERRORSHIFT;
  }

private:

  const unsigned int * word_;

};




/** \class DTROBHeaderWord
 *  DT ROB Header interpreter. 
 *  It interprets the ROB_ID (5 bits), the Event ID (12 bits) 
 *  and the Bunch ID (12 bits).
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTROBHeaderWord {

public:

  /// Constructor
  DTROBHeaderWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTROBHeaderWord() {}

  int robID() { return (*word_ & ROB_ID_MASK) >> WORDTYPESHIFT;} 
  int eventID() { return (*word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;} 
  int bunchID() { return (*word_ & BUNCH_ID_MASK);} 


  static void set(unsigned char* word,
		  int rob_id,
		  int event_id,
		  int bunch_id) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::headerControlWord << WORDCONTROLSHIFT |
      rob_id << WORDTYPESHIFT |
      event_id << EVENT_ID_SHIFT |
      bunch_id;
  }
  

private:

  const unsigned int * word_;

};


/** \class DTROBTrailerWord
 *  DT ROB Trailer interpreter. 
 *  It interprets the ROB_ID (5 bits), the Event ID (12 bits) 
 *  and the Word ID (12 bits).
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTROBTrailerWord {

public:

  /// Constructor
  DTROBTrailerWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTROBTrailerWord() {}

  int robID() { return (*word_ & ROB_ID_MASK) >> WORDTYPESHIFT;} 
  int eventID() { return (*word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;} 
  int wordCount() { return (*word_ & WORD_COUNT_MASK);} 

  static void set(unsigned char* word,
		  int rob_id,
		  int event_id,
		  int word_count) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::trailerControlWord << WORDCONTROLSHIFT |
      rob_id << WORDTYPESHIFT |
      event_id << EVENT_ID_SHIFT |
      word_count;
  }
  

private:

  const unsigned int * word_;
};




/** \class DTTDCHeaderWord
 *  DT TDC Header interpreter. 
 *  It interprets the Parity Checks, FIFO occupancy, Lokeced channels (all 1 bit),
 *  the TDC_ID (2 bits), the Event ID (12 bits) and the Bunch ID (12 bits).
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTTDCHeaderWord {

public:

  /// Constructor
  DTTDCHeaderWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTTDCHeaderWord() {}

  int PC() { return (*word_ & PC_MASK) >> PC_SHIFT;} 
  int PAF() { return (*word_ & PAF_MASK) >> PAF_SHIFT;} 
  int HU() { return (*word_ & HU_MASK) >> HU_SHIFT;} 
  int tdcID() { return (*word_ & TDC_ID_MASK) >> TDC_ID_SHIFT;} 
  int eventID() { return (*word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;} 
  int bunchID() { return (*word_ & BUNCH_ID_MASK);} 

  static void set(unsigned char* word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int event_id,
		  int bunch_id) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::tdcHeaderControlWord << WORDCONTROLSHIFT |
      pc << PC_SHIFT |
      paf << PAF_SHIFT |
      hu << HU_SHIFT |
      tdc_id << TDC_ID_SHIFT |
      event_id << EVENT_ID_SHIFT |
      bunch_id;
  }


private:

  const unsigned int * word_;
};


/** \class DTTDCTrailerWord
 *  DT TDC Trailer interpreter. 
 *  It interprets the Parity Checks, FIFO occupancy, Lokeced channels (all 1 bit),
 *  the TDC_ID (2 bits), the Event ID (12 bits) and the Word ID (12 bits).
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTTDCTrailerWord {

public:

  /// Constructor
  DTTDCTrailerWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTTDCTrailerWord() {}

  int PC() { return (*word_ & PC_MASK) >> PC_SHIFT;} 
  int PAF() { return (*word_ & PAF_MASK) >> PAF_SHIFT;} 
  int HU() { return (*word_ & HU_MASK) >> HU_SHIFT;} 
  int tdcID() { return (*word_ & TDC_ID_MASK) >> TDC_ID_SHIFT;} 
  int eventID() { return (*word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;} 
  int wordCount() { return (*word_ & WORD_COUNT_MASK);} 

  static void set(unsigned char* word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int event_id,
		  int word_count) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::tdcTrailerControlWord << WORDCONTROLSHIFT |
      pc << PC_SHIFT |
      paf << PAF_SHIFT |
      hu << HU_SHIFT |
      tdc_id << TDC_ID_SHIFT |
      event_id << EVENT_ID_SHIFT |
      word_count;
  }

private:

  const unsigned int * word_;
};


/** \class DTTDCMeasurementWord
 *  DT TDC Measurement interpreter. 
 *  It interprets the Parity Checks, FIFO occupancy, Lokeced channels (all 1 bit),
 *  the TDC_ID (2 bits), the TDC channel (5 bits), and the TDC time (19 bits)
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTTDCMeasurementWord {

public:

  /// Constructor
  DTTDCMeasurementWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTTDCMeasurementWord() {}

  int PC() { return (*word_ & PC_MASK) >> PC_SHIFT;} 
  int PAF() { return (*word_ & PAF_MASK) >> PAF_SHIFT;} 
  int HU() { return (*word_ & HU_MASK) >> HU_SHIFT;} 
  int tdcID() { return (*word_ & TDC_ID_MASK) >> TDC_ID_SHIFT;} 
  int tdcChannel() { return (*word_ & TDC_CHANNEL_MASK) >> TDC_CHANNEL_SHIFT;} 
  int tdcTime() { return (*word_ & TDC_TIME_MASK);} 


  static void set(unsigned char* word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int tdc_channel,
		  int tdc_time) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::tdcDataControlWord << WORDCONTROLSHIFT |
      pc << PC_SHIFT |
      paf << PAF_SHIFT |
      hu << HU_SHIFT |
      tdc_id << TDC_ID_SHIFT |
      tdc_channel << TDC_CHANNEL_SHIFT |
      tdc_time;
  }



private:

  const unsigned int * word_;
};


/** \class DTTDCErrorWord
 *  DT TDC Error interpreter. 
 *  It interprets the Parity Checks, FIFO occupancy, Lokeced channels (all 1 bit),
 *  the TDC_ID (2 bits) and the TDC error flag (15 bits)
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTTDCErrorWord {

public:

  /// Constructor
  DTTDCErrorWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTTDCErrorWord() {}

  int PC() { return (*word_ & PC_MASK) >> PC_SHIFT;} 
  int PAF() { return (*word_ & PAF_MASK) >> PAF_SHIFT;} 
  int HU() { return (*word_ & HU_MASK) >> HU_SHIFT;} 
  int tdcID() { return (*word_ & TDC_ID_MASK) >> TDC_ID_SHIFT;} 
  int tdcError() { return (*word_ & TDC_ERROR_MASK);} 

  static void set(unsigned char* word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int tdc_error) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::errorControlWord << WORDCONTROLSHIFT |
      pc << PC_SHIFT |
      paf << PAF_SHIFT |
      hu << HU_SHIFT |
      tdc_id << TDC_ID_SHIFT |
      0 << ERRORSHIFT |
      tdc_error;
  }

private:

  const unsigned int * word_;
};


/** \class DTLocalTriggerHeaderWord
 *  DT Sector Collector header interpreter. 
 *  It interprets ROS event ID (12 bits) and the Sector Collector FIFO occupancy (8 bits)
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerHeaderWord {

public:

  /// Constructor
  DTLocalTriggerHeaderWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTLocalTriggerHeaderWord() {}

  int EventID() { return (*word_ & EVENT_ID_MASK) >> EVENT_ID_SHIFT;}
  int SCFO() { return (*word_ & SCFO_MASK);}

  
  static void set(unsigned char* word,
		  int event_id,
		  int scfo) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::headerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord << WORDTYPESHIFT |
      event_id << EVENT_ID_SHIFT |
      scfo; 
  }

private:

  const unsigned int * word_;
};


/** \class DTLocalTriggerTrailerWord
 *  DT Sector Collector trailer interpreter. 
 *  It interprets the word count (16 bits)
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerTrailerWord {

public:

  /// Constructor
  DTLocalTriggerTrailerWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTLocalTriggerTrailerWord() {}

  int wordCount() { return (*word_ & TRIGGER_WORD_COUNT_MASK);}

  static void set(unsigned char* word,
		  int word_count) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::trailerControlWord << WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord << WORDTYPESHIFT |
      word_count; 
  }


private:

  const unsigned int * word_;
};


/** \class DTLocalTriggerDataWord
 *  DT Sector Collector data interpreter. 
 *  It interprets the Sector Collector data (16 bits)
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerDataWord {

public:

  /// Constructor
  DTLocalTriggerDataWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTLocalTriggerDataWord() {}

  int SCData() { return (*word_ & TRIGGER_DATA_MASK);}

  static void set(unsigned char* word,
		  int sc_data) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::tdcDataControlWord << WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord << WORDTYPESHIFT |
      sc_data; 
  }


private:

  const unsigned int * word_;
};


/** \class DTDDUFirstStatusWord
 *  DT DDU status 1 interpreter (8 bits word). 
 *  It interprets the error messages from each DDU channel
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTDDUFirstStatusWord {

public:

  /// Constructor
  DTDDUFirstStatusWord(const unsigned int* index) : 
    word_(index) {}

  /// Destructor
  virtual ~DTDDUFirstStatusWord() {}

  int channelEnabled() { return (*word_ & 0x1);}
  int timeout() { return (*word_ & 0x2) >> 1;}
  int eventTrailerLost() { return (*word_ & 0x4) >> 2;}
  int opticalFiberSignalLost() { return (*word_ & 0x8) >> 3;}
  int tlkPropagationError() { return (*word_ & 0x10) >> 4;}
  int tlkPatternError() { return (*word_ & 0x20) >> 5;}
  int tlkSignalLost() { return (*word_ & 0x40) >> 6;}
  int errorFromROS() { return (*word_ & 0x80) >> 7;}


private:

  const unsigned int * word_;
};


/** \class DTDDUSecondStatusWord
 *  DT DDU status 2 interpreter. 
 *  It interprets the (16 bits)
 *  WARNING!! : It interprets the second part of a 64 bits word!
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 */
class DTDDUSecondStatusWord {
  
public:
  
  /// Constructor
  DTDDUSecondStatusWord(const unsigned int* index) : 
    word_(index) {}
  
  /// Destructor
  virtual ~DTDDUSecondStatusWord() {}
  
  int l1AIDError() { return (*word_ & 0x1); }
  int bxIDError() { return (*word_ & 0x2) >> 1; }
  int fifoFull() { return (*word_ & 0x1C ) >> 2; }
  int inputFifoFull() { return (*word_ & 0xE0) >> 5; }
  int fifoAlmostFull() { return (*word_ & 0x700) >> 8; }
  int inputFifoAlmostFull() { return (*word_ & 0x3800) >> 11; }
  int fifoAlmostEmpty() { return (*word_ & 0x1C000) >> 14; }
  int inputFifoAlmostEmpty() { return (*word_ & 0xE0000) >> 17; }
  int outputFifoFull() { return (*word_ & 0x100000) >> 20; }
  int outputFifoAlmostFull() { return (*word_ & 0x200000) >> 21; }
  int outputFifoAlmostEmpty() { return (*word_ & 0x400000) >> 22; }

private:
  
  const unsigned int * word_;
};




#endif
