#include "EventFilter/Utilities/interface/GlobalEventNumber.h"
#include "interface/shared/fed_header.h" // from xdaq

namespace evf{
  namespace evtn{

    unsigned int offset(bool evm)
    {
      if(evm)
	return sizeof(fedh_t) + (EVM_GTFE_BLOCK*2 + EVM_TCS_TRIGNR_OFFSET) * SLINK_WORD_SIZE / 2;
      else
	return sizeof(fedh_t) + DAQ_TOTTRG_OFFSET * SLINK_WORD_SIZE / 2;
    }
    unsigned int get(const unsigned char *p, bool evm)
    {
      return *(unsigned int*)( p+offset(evm) );
    }
  }
}
