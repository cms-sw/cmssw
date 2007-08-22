#include "interface/shared/fed_header.h" // from xdaq
namespace evf{
  namespace evtn{
      const unsigned int SLINK_WORD_SIZE = 8;
      const unsigned int DAQ_TOTTRG_OFFSET = 2; //offset in 32-bit words
      const unsigned int DAQ_BOARDID_OFFSET = 1;
      const unsigned int DAQ_BOARDID_MASK = 0xffff0000;
      const unsigned int DAQ_BOARDID_SHIFT = 24;
      const unsigned int DAQ_BOARDID_VALUE = 0x11;
      const unsigned int EVM_GTFE_BLOCK = 6; //size in 64-bit words
      const unsigned int EVM_TCS_TRIGNR_OFFSET = 5; //offset in 32-bit words
      unsigned int offset(bool);
      inline bool daq_board_sense(const unsigned char *p)
	{
	  return (*(unsigned int*)(p + sizeof(fedh_t) + DAQ_BOARDID_OFFSET * SLINK_WORD_SIZE / 2) >> DAQ_BOARDID_SHIFT) == DAQ_BOARDID_VALUE;
	}
      unsigned int get(const unsigned char *, bool);
  }
}
