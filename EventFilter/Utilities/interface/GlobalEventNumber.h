namespace evf{
  namespace evtn{
      const unsigned int SLINK_WORD_SIZE = 8;
      const unsigned int DAQ_TOTTRG_OFFSET = 2; //offset in 32-bit words
      const unsigned int EVM_GTFE_BLOCK = 6; //size in 64-bit words
      const unsigned int EVM_TCS_TRIGNR_OFFSET = 5; //offset in 32-bit words
      unsigned int offset(bool);
      unsigned int get(const unsigned char *, bool);
  }
}
