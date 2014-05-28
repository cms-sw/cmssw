#ifndef CSCCFEBTimeSlice_h
#define CSCCFEBTimeSlice_h


/**
 CFEB Data Stream 
The ordering of the words in the data stream from a single CFEB is described by the following nested loops: 

do (N_ts  time samples){ 
  do (Gray code loop over 16 CSC Strips; S=0,1,3,2,6,7,5,4,12,13,15,14,10,11,9,8){ 
    do (loop over 6 CSC Layers; L=3,1,5,6,4,2){ 
    }
  } 
  CRC word 
  CFEB Info word 98 
  CFEB Info word 99 
  Dummy word (0x7FFF)
}
*/

struct CSCCFEBDataWord {
  unsigned short adcCounts   : 12;
  unsigned short adcOverflow : 1;
  /// combined from all 16 strips to make a word
  unsigned short controllerData : 1;
  /// Overlapped sample flag (normally HIGH;
  /// set LOW when two separate LCTs share a time sample).
  unsigned short overlappedSampleFlag : 1;
  /// Always LOW for data words.  HIGH for DDU Code word
  /// (maybe End of Event or Error)
  unsigned short errorstat : 1;
};

#include <iostream>
#include <string.h> //for bzero

struct CSCCFEBSCAControllerWord {
  /**
TRIG_TIME indicates which of the eight time samples in the 400ns SCA block (lowest bit is the first sample, highest bit the eighth sample) corresponds to the arrival of the LCT; it should be at some fixed phase relative to the peak of the CSC pulse.  SCA_BLK is the SCA Capacitor block used for this time sample. L1A_PHASE and LCT_PHASE show the phase of the 50ns CFEB digitization clock at the time the trigger was received (1=clock high, 0=clock low).  SCA_FULL indicates lost SCA data due to SCA full condition.  The TS_FLAG bit indicates the number of time samples to digitize per event; high=16 time samples, low=8 time samples. 

  */
  explicit CSCCFEBSCAControllerWord(unsigned short frame);
  CSCCFEBSCAControllerWord() {bzero(this, 2);}

  unsigned short trig_time : 8;
  unsigned short sca_blk   : 4;
  unsigned short l1a_phase : 1;
  unsigned short lct_phase : 1;
  unsigned short sca_full  : 1;
  unsigned short ts_flag   : 1;
};




class CSCCFEBTimeSlice {
 public:
  CSCCFEBTimeSlice();

  /// input from 0 to 95
  CSCCFEBDataWord * timeSample(int index) const {
    return (CSCCFEBDataWord *)(theSamples+index);
  }

  /// layer and element count from one
  // CSCCFEBDataWord * timeSample(int layer, int channel) const;

  /// !!! Important change. Use isDCFEB flag in user code to distinguish between CFEB and DCFEB
  /// !!! Use CSCCFEBData::isDCFEB() function to get this flag from CSCCFEBData object
  CSCCFEBDataWord * timeSample(int layer, int channel, bool isDCFEB=false) const;

  /// whether we keep 8 or 16 time samples
  bool sixteenSamples() {/*return scaControllerWord(1).ts_flag;i*/
    return timeSample(95)->controllerData;}
  unsigned sizeInWords() const {return 100;}


  /// unpacked from the controller words for each channel in the layer
  CSCCFEBSCAControllerWord scaControllerWord(int layer) const ;
  
  void setControllerWord(const CSCCFEBSCAControllerWord & controllerWord);

  /// Old CFEB format: dummy word 100 should be 0x7FFF
  /// New CFEB format: the sum of word 97 and 100 should be 0x7FFF (word 100 is inverted word 97)
  bool check() const {return ((dummy == 0x7FFF)||((dummy+crc)== 0x7FFF));}

  bool checkCRC() const {return crc==calcCRC();}
  
  unsigned calcCRC() const;

  /// =VB= Set calculated CRC value for simulated CFEB Time Slice data
  void setCRC() { crc=calcCRC(); dummy=0x7FFF-crc;}

  friend std::ostream & operator<<(std::ostream & os, const CSCCFEBTimeSlice &);


  ///accessors for words 97, 98 and 99
  unsigned  get_crc()               const {return crc;}
  unsigned  get_n_free_sca_blocks() const {return n_free_sca_blocks;}
  unsigned  get_lctpipe_count()     const {return lctpipe_count;}
  unsigned  get_lctpipe_full()      const {return lctpipe_full;}
  unsigned  get_l1pipe_full()       const {return l1pipe_full;}
  unsigned  get_lctpipe_empty()     const {return lctpipe_empty;}
  unsigned  get_l1pipe_empty()      const {return l1pipe_empty;}
  unsigned  get_buffer_warning()    const {return buffer_warning;}
  unsigned  get_buffer_count()      const {return buffer_count;}
  unsigned  get_L1A_number()        const {return L1A_number;}

  void 	    set_L1Anumber(unsigned l1a) {L1A_number = l1a & 0x3F;}

 private:
  unsigned short theSamples[96];

  /// WORD 97
  unsigned crc : 16;

  /// WORD 98
  unsigned n_free_sca_blocks : 4;
  unsigned lctpipe_count : 4;
  unsigned lctpipe_full  : 1;
  unsigned l1pipe_full   : 1;
  unsigned lctpipe_empty : 1;
  unsigned l1pipe_empty  : 1;
  unsigned blank_space_1 : 4;


  /// WORD 99
  unsigned buffer_warning : 1;
  unsigned buffer_count : 5;
  unsigned L1A_number :6;
  unsigned blank_space_3 : 4; 

  /// WORD 100
  unsigned dummy : 16;
};

#endif
