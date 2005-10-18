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


/** \class DTROSWordType
 *  Enumeration of DT Read Out Sector (ROS) word types.
 *
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTROSWordType {

public:
  
  /// Constructor
  DTROSWordType(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {};

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
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == tdcHeaderControlWord ) wordType = TDCHeader;

    // TDC Trailer
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == tdcTrailerControlWord ) wordType = TDCTrailer;

    // TDC Measurement
    if ( ((*word_ & WORDTYPEMASK) >> WORDCONTROLSHIFT) == tdcDataControlWord ) wordType = TDCMeasurement;

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
  void update() { word_ += DTDDU_WORD_SIZE; }


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
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTROSHeaderWord {

public:

  DTROSHeaderWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {} 

  int TTCEventCounter() { return  *word_ & TTC_EVENT_COUNTER_MASK; }

  static void set(unsigned char* word,
		  int ttc_event_counter) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::headerControlWord >> WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord >> WORDTYPESHIFT |
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
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTROSTrailerWord {

public:

  DTROSTrailerWord(const unsigned char* index) : 
    word_((const unsigned int*)(index)) {} 

  int TFF() { return *word_ & TFF_MASK; }
  int TPX() { return *word_ & TPX_MASK; }
  int ECHO() { return *word_ & ECHO_MASK; }
  int ECLO() { return *word_ & ECLO_MASK; }
  int BCO() { return *word_ & BCO_MASK; }
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
      DTROSWordType::trailerControlWord >> WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord >> WORDTYPESHIFT |
      tff >> TFF_SHIFT |
      tpx >> TPX_SHIFT |
      echo >> ECHO_SHIFT |
      bco >> BCO_SHIFT |
      event_word_count;
  }


private:

  const unsigned int * word_;

};


/** \class DTROSErrorWord
 *  DT ROS Error interpreter. 
 *  It interprets the Error type and the ROB_ID (2 bits) 
 *
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTROSErrorWord {

public:

  DTROSErrorWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int errorType() { return *word_ & ERROR_TYPE_MASK;} 
  int robID() { return *word_ & ERROR_ROB_ID_MASK;} 

  static void set(unsigned char* word,
		  int error_type,
		  int rob_id) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::errorControlWord >> WORDCONTROLSHIFT |
      DTROSWordType::rosTypeWord >> WORDTYPESHIFT |
      error_type >> ERROR_TYPE_SHIFT |
      rob_id >> ERROR_ROB_ID_SHIFT |
      1 >> ERRORSHIFT;
  }

private:

  const unsigned int * word_;

};




/** \class DTROBHeaderWord
 *  DT ROB Header interpreter. 
 *  It interprets the ROB_ID (5 bits), the Event ID (12 bits) 
 *  and the Bunch ID (12 bits).
 *
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTROBHeaderWord {

public:

  DTROBHeaderWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int robID() { return *word_ & ROB_ID_MASK;} 
  int eventID() { return *word_ & EVENT_ID_MASK;} 
  int bunchID() { return *word_ & BUNCH_ID_MASK;} 


  static void set(unsigned char* word,
		  int rob_id,
		  int event_id,
		  int bunch_id) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::headerControlWord >> WORDCONTROLSHIFT |
      rob_id >> WORDTYPESHIFT |
      event_id >> EVENT_ID_SHIFT |
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
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTROBTrailerWord {

public:

  DTROBTrailerWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int robID() { return *word_ & ROB_ID_MASK;} 
  int eventID() { return *word_ & EVENT_ID_MASK;} 
  int wordCount() { return *word_ & WORD_COUNT_MASK;} 

  static void set(unsigned char* word,
		  int rob_id,
		  int event_id,
		  int word_count) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::trailerControlWord >> WORDCONTROLSHIFT |
      rob_id >> WORDTYPESHIFT |
      event_id >> EVENT_ID_SHIFT |
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
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTTDCHeaderWord {

public:

  DTTDCHeaderWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int PC() { return *word_ & PC_MASK;} 
  int PAF() { return *word_ & PAF_MASK;} 
  int HU() { return *word_ & HU_MASK;} 
  int tdcID() { return *word_ & TDC_ID_MASK;} 
  int eventID() { return *word_ & EVENT_ID_MASK;} 
  int bunchID() { return *word_ & BUNCH_ID_MASK;} 

  static void set(unsigned char* word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int event_id,
		  int bunch_id) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::tdcHeaderControlWord >> WORDCONTROLSHIFT |
      pc >> PC_SHIFT |
      paf >> PAF_SHIFT |
      hu >> HU_SHIFT |
      tdc_id >> TDC_ID_SHIFT |
      event_id >> EVENT_ID_SHIFT |
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
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTTDCTrailerWord {

public:

  DTTDCTrailerWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int PC() { return *word_ & PC_MASK;} 
  int PAF() { return *word_ & PAF_MASK;} 
  int HU() { return *word_ & HU_MASK;} 
  int tdcID() { return *word_ & TDC_ID_MASK;} 
  int eventID() { return *word_ & EVENT_ID_MASK;} 
  int wordCount() { return *word_ & WORD_COUNT_MASK;} 

  static void set(unsigned char* word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int event_id,
		  int word_count) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::tdcTrailerControlWord >> WORDCONTROLSHIFT |
      pc >> PC_SHIFT |
      paf >> PAF_SHIFT |
      hu >> HU_SHIFT |
      tdc_id >> TDC_ID_SHIFT |
      event_id >> EVENT_ID_SHIFT |
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
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTTDCMeasurementWord {

public:

  DTTDCMeasurementWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int PC() { return *word_ & PC_MASK;} 
  int PAF() { return *word_ & PAF_MASK;} 
  int HU() { return *word_ & HU_MASK;} 
  int tdcID() { return *word_ & TDC_ID_MASK;} 
  int tdcChannel() { return *word_ & TDC_CHANNEL_MASK;} 
  int tdcTime() { return *word_ & TDC_TIME_MASK;} 


  static void set(unsigned char* word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int tdc_channel,
		  int tdc_time) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::tdcDataControlWord >> WORDCONTROLSHIFT |
      pc >> PC_SHIFT |
      paf >> PAF_SHIFT |
      hu >> HU_SHIFT |
      tdc_id >> TDC_ID_SHIFT |
      tdc_channel >> TDC_CHANNEL_SHIFT |
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
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTTDCErrorWord {

public:

  DTTDCErrorWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int PC() { return *word_ & PC_MASK;} 
  int PAF() { return *word_ & PAF_MASK;} 
  int HU() { return *word_ & HU_MASK;} 
  int tdcID() { return *word_ & TDC_ID_MASK;} 
  int tdcError() { return *word_ & TDC_ERROR_MASK;} 

  static void set(unsigned char* word,
		  int pc,
		  int paf,
		  int hu,
		  int tdc_id,
		  int tdc_error) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::errorControlWord >> WORDCONTROLSHIFT |
      pc >> PC_SHIFT |
      paf >> PAF_SHIFT |
      hu >> HU_SHIFT |
      tdc_id >> TDC_ID_SHIFT |
      0 >> ERRORSHIFT |
      tdc_error;
  }

private:

  const unsigned int * word_;
};


/** \class DTLocalTriggerHeaderWord
 *  DT TDC Error interpreter. 
 *  It interprets ROS event ID (12 bits) and the Sector Collector FIFO occupancy (8 bits)
 *
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerHeaderWord {

public:

  DTLocalTriggerHeaderWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int EventID() { return *word_ & EVENT_ID_MASK;}
  int SCFO() { return *word_ & SCFO_MASK;}

  
  static void set(unsigned char* word,
		  int event_id,
		  int scfo) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::headerControlWord >> WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord >> WORDTYPESHIFT |
      event_id >> EVENT_ID_SHIFT |
      scfo; 
  }

private:

  const unsigned int * word_;
};


/** \class DTLocalTriggerTrailerWord
 *  DT TDC Error interpreter. 
 *  It interprets the word count (16 bits)
 *
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerTrailerWord {

public:

  DTLocalTriggerTrailerWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int wordCount() { return *word_ & TRIGGER_WORD_COUNT_MASK;}

  static void set(unsigned char* word,
		  int word_count) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::trailerControlWord >> WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord >> WORDTYPESHIFT |
      word_count; 
  }


private:

  const unsigned int * word_;
};


/** \class DTLocalTriggerDataWord
 *  DT TDC Error interpreter. 
 *  It interprets the Sector Collector data (16 bits)
 *
 *  $Date: 2005/10/07 09:24:10 $
 *  $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 */
class DTLocalTriggerDataWord {

public:

  DTLocalTriggerDataWord(const unsigned char* index) : 
    word_(reinterpret_cast<const unsigned int*>(index)) {}

  int SCData() { return *word_ & TRIGGER_DATA_MASK;}

  static void set(unsigned char* word,
		  int sc_data) {
    unsigned int* iword = reinterpret_cast<unsigned int*> (word);
    
    *iword = 
      DTROSWordType::tdcDataControlWord >> WORDCONTROLSHIFT |
      DTROSWordType::scTypeWord >> WORDTYPESHIFT |
      sc_data; 
  }


private:

  const unsigned int * word_;
};



#endif
