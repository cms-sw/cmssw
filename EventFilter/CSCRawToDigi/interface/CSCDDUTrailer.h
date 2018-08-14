#ifndef CSCDDUTrailer_h
#define CSCDDUTrailer_h

#include <iostream>
#include <cstdint>
#include <cstring> // bzero
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"

/** documented at  http://www.physics.ohio-state.edu/~cms/ddu/ddu2.html
 */



class CSCDDUTrailer {

 public:

  CSCDDUTrailer() 
    {
      bzero(this, sizeInWords()*2);
      trailer2_1 = trailer2_2 = trailer2_4 = 0x8000;
      trailer2_3 = 0xFFFF;
    }
  CSCDDUTrailer(const CSCDDUStatusDigi & digi)
    {
      memcpy(this, digi.trailer(), sizeInWords()*2);
    }
  
  void setFromBuffer(uint16_t const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  static unsigned sizeInWords() {return 12;}
  
  bool check() const {
    //std::cout << std:: hex << "DDUTRAILER CHECK " << trailer2_1 << " " 
    //      << trailer2_2  << " " << trailer2_3 << " " 
    //      << trailer2_4 << std:: dec << std::endl;
    return trailer2_1 == 0x8000 && trailer2_2 == 0x8000
                   && trailer2_3 == 0xFFFF && trailer2_4 == 0x8000;}

  unsigned short * data() {return (unsigned short *) this;}
  
  //These are accessors to use for calling private members    
  
  unsigned errorstat() const { return errorstat_; }
  unsigned wordcount() const { return word_count_; }  
  void setWordCount(unsigned wordcount) {word_count_ = wordcount;}
  //@@ Tim: This seems wrong to me so just pull it
  //@@  void setDMBDAV(int dmbId) {dmb_full_ |= (1 << dmbId);}
  unsigned dmb_warn() const { return dmb_warn_; }  
  unsigned dmb_full() const { return dmb_full_; }
  unsigned reserved() const { return whatever; } 

  
  
 private:
  
  /// should be 8000/8000/FFFF/8000
  unsigned trailer2_1 : 16;
  unsigned trailer2_2 : 16;
  unsigned trailer2_3 : 16;
  unsigned trailer2_4 : 16;

  /// Active DMB Count (4 bits)
  unsigned dmb_warn_   : 16;
  unsigned dmb_full_   : 16;
  unsigned errorstat_  : 32;

  // DDU2004
  //  unsigned reserved_bits     : 4;
  //unsigned ddu_tts_status    : 4;
  //unsigned event_status      : 8;
  //unsigned event_crc         : 16;
  //

  //DDU2000
  //unsigned s_link_status  : 4;
  //unsigned crc_check_word : 16;
  //unsigned whatever       : 4;
  //unsigned event_status   : 8;
  //

  //the following bits change their meaning in DDU2004
  unsigned word1 : 4;
  unsigned word2 : 4;
  unsigned word3 : 4;
  unsigned word4 : 4;

  unsigned word5 : 4;
  unsigned word6 : 4;
  unsigned word7 : 4;
  unsigned word8 : 4;


  /// in 64-bit words
  /// DDU_WC = (6 + 25*N_ts*nCFEB + 3*nDMB)
  unsigned word_count_        : 24;
  unsigned whatever           : 4;
  ///constant, should be 1010
  unsigned cms_directive_0xA  : 4;

};
#endif
