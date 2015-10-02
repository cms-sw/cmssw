#ifndef CTP7Format_hh
#define CTP7Format_hh

class CTP7Format {
 public:
  CTP7Format():
    EVENT_HEADER_WORDS(6),
    CHANNEL_HEADER_WORDS(2),
    CHANNEL_DATA_WORDS_PER_BX(6),
    NLINKS(36){
  }

  const uint32_t EVENT_HEADER_WORDS;
  const uint32_t CHANNEL_HEADER_WORDS;
  const uint32_t CHANNEL_DATA_WORDS_PER_BX;
  //const uint32_t NIntsBRAMDAQ;
  const uint32_t NLINKS;
  
 private:

};

#endif
