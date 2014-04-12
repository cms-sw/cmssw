#include "EventFilter/FEDInterface/interface/GlobalEventNumber.icc"

namespace evf{
  namespace evtn{


    bool evm_board_sense(const unsigned char *p, size_t size)
    {
      switch(size){
      case BST32_3BX:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0000;
	EVM_FDL_NOBX = 3;
	break;
      case BST32_5BX:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0000;
	EVM_FDL_NOBX = 5;
	break;
      case BST52_3BX:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0011;
	EVM_FDL_NOBX = 3;
	break;
      case BST52_5BX:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0011;
	EVM_FDL_NOBX = 5;
	break;
      default:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0000;
	EVM_FDL_NOBX = 3;	    
      }
      return (*(const unsigned int*)(p + sizeof(fedh_t) + EVM_BOARDID_OFFSET * SLINK_WORD_SIZE / 2) >> EVM_BOARDID_SHIFT) == EVM_BOARDID_VALUE;
    }


    void evm_board_setformat(size_t size)
    {
      switch(size){
      case BST32_3BX:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0000;
	EVM_FDL_NOBX = 3;
	break;
      case BST32_5BX:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0000;
	EVM_FDL_NOBX = 5;
	break;
      case BST52_3BX:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0011;
	EVM_FDL_NOBX = 3;
	break;
      case BST52_5BX:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0011;
	EVM_FDL_NOBX = 5;
	break;
      default:
	EVM_GTFE_BLOCK = EVM_GTFE_BLOCK_V0000;
	EVM_FDL_NOBX = 3;	    
      }
    }
    

    unsigned int offset(bool evm)
    {
      if(evm)
	return sizeof(fedh_t) + (EVM_GTFE_BLOCK*2 + EVM_TCS_TRIGNR_OFFSET) * SLINK_WORD_SIZE / 2;
      else
	return sizeof(fedh_t) + DAQ_TOTTRG_OFFSET * SLINK_WORD_SIZE / 2;
    }
    unsigned int get(const unsigned char *p, bool evm)
    {
      return *(const unsigned int*)( p+offset(evm) );
    }
    unsigned int gtpe_get(const unsigned char *p)
    {
      return *(const unsigned int*)( p + GTPE_TRIGNR_OFFSET*SLINK_HALFWORD_SIZE );
    }
    unsigned int getlbn(const unsigned char *p)
    { 
      return (*(const unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK*2 + EVM_TCS_LSBLNR_OFFSET) * SLINK_WORD_SIZE / 2)) 
	& EVM_TCS_LSBLNR_MASK;
    }
    unsigned int gtpe_getlbn(const unsigned char *p)
    { 
      return gtpe_getorbit(p)/0x00100000;
    }
    unsigned int getgpslow(const unsigned char *p)
    { 
      return (*(const unsigned int*)( p+sizeof(fedh_t) + EVM_GTFE_BSTGPS_OFFSET * SLINK_WORD_SIZE / 2));
    }
    unsigned int getgpshigh(const unsigned char *p)
    { 
      return (*(const unsigned int*)( p+sizeof(fedh_t) + EVM_GTFE_BSTGPS_OFFSET * SLINK_WORD_SIZE / 2 + SLINK_HALFWORD_SIZE));
    }
    unsigned int getorbit(const unsigned char *p)
    { 
      return (*(const unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK*2 + EVM_TCS_ORBTNR_OFFSET) * SLINK_WORD_SIZE / 2));
    }
    unsigned int getevtyp(const unsigned char *p)
    { 
      return (((*(const unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK*2 + EVM_TCS_LSBLNR_OFFSET) * SLINK_WORD_SIZE / 2)) 
	      & EVM_TCS_EVNTYP_MASK) >> EVM_TCS_EVNTYP_SHIFT);
    }
    unsigned int gtpe_getorbit(const unsigned char *p)
    { 
      return (*(const unsigned int*)( p + GTPE_ORBTNR_OFFSET * SLINK_HALFWORD_SIZE));
    }
    unsigned int getfdlbx(const unsigned char *p)
    { 
      return (*(const unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK + EVM_TCS_BLOCK 
						    + EVM_FDL_BLOCK * (EVM_FDL_NOBX/2) ) * SLINK_WORD_SIZE +
				EVM_FDL_BCNRIN_OFFSET * SLINK_HALFWORD_SIZE)) &  EVM_TCS_BCNRIN_MASK;
    }
    unsigned int gtpe_getbx(const unsigned char *p)
    { 
      return (*(const unsigned int*)( p + GTPE_BCNRIN_OFFSET * SLINK_HALFWORD_SIZE)) &  GTPE_BCNRIN_MASK;
    }
    unsigned int getfdlpsc(const unsigned char *p)
    { 
      return (*(const unsigned int*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK + EVM_TCS_BLOCK 
						    + EVM_FDL_BLOCK * (EVM_FDL_NOBX/2)) * SLINK_WORD_SIZE +
				EVM_FDL_PSCVSN_OFFSET * SLINK_HALFWORD_SIZE));
    }
    unsigned long long getfdlttr(const unsigned char *p)
    {
       return (*(const unsigned long long*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK + EVM_TCS_BLOCK 
							   + EVM_FDL_BLOCK * (EVM_FDL_NOBX/2)) * SLINK_WORD_SIZE +
				EVM_FDL_TECTRG_OFFSET * SLINK_HALFWORD_SIZE));
    }
    unsigned long long getfdlta1(const unsigned char *p)
    {
       return (*(const unsigned long long*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK + EVM_TCS_BLOCK 
							   + EVM_FDL_BLOCK * (EVM_FDL_NOBX/2)) * SLINK_WORD_SIZE +
				EVM_FDL_ALGOB1_OFFSET * SLINK_HALFWORD_SIZE));
    }
    unsigned long long getfdlta2(const unsigned char *p)
    {
       return (*(const unsigned long long*)( p+sizeof(fedh_t) + (EVM_GTFE_BLOCK + EVM_TCS_BLOCK 
							   + EVM_FDL_BLOCK * (EVM_FDL_NOBX/2)) * SLINK_WORD_SIZE +
				EVM_FDL_ALGOB2_OFFSET * SLINK_HALFWORD_SIZE));
    }
  }
}
