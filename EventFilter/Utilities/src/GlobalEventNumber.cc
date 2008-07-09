#include "EventFilter/Utilities/interface/GlobalEventNumber.h"

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
    unsigned int getlbn(const unsigned char *p)
    { 
      return (*(unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK*2 + EVM_TCS_LSBLNR_OFFSET) * SLINK_WORD_SIZE / 2)) 
	& EVM_TCS_LSBLNR_MASK;
    }
    unsigned int getgpslow(const unsigned char *p)
    { 
      return (*(unsigned int*)( p+sizeof(fedh_t) + EVM_GTFE_BSTGPS_OFFSET * SLINK_WORD_SIZE / 2));
    }
    unsigned int getgpshigh(const unsigned char *p)
    { 
      return (*(unsigned int*)( p+sizeof(fedh_t) + EVM_GTFE_BSTGPS_OFFSET * SLINK_WORD_SIZE / 2 + SLINK_HALFWORD_SIZE));
    }
    unsigned int getorbit(const unsigned char *p)
    { 
      return (*(unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK*2 + EVM_TCS_ORBTNR_OFFSET) * SLINK_WORD_SIZE / 2));
    }
    unsigned int getfdlbx(const unsigned char *p)
    { 
      return (*(unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK + EVM_TCS_BLOCK + EVM_FDL_BLOCK) * SLINK_WORD_SIZE +
				EVM_FDL_BCNRIN_OFFSET * SLINK_HALFWORD_SIZE)) &  EVM_TCS_BCNRIN_MASK;
    }

    unsigned int getfdlpsc(const unsigned char *p)
    { 
      return (*(unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK + EVM_TCS_BLOCK + EVM_FDL_BLOCK) * SLINK_WORD_SIZE +
				EVM_FDL_PSCVSN_OFFSET * SLINK_HALFWORD_SIZE));
    }

  }
}
